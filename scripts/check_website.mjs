// Checks the running website in headless Chrome, the way a person uses it, and saves screenshots.
//
//   scripts/start_local.sh                      (in another terminal: everything running)
//   node scripts/check_website.mjs [http://127.0.0.1:8000/]
//
// It clicks the examples, searches, opens a control's text, asks for cited answers (one covered,
// one not), and checks the layout on a desktop and a phone, in light and dark mode. Nothing is
// installed: it drives the Chrome already on this Mac through its DevTools protocol.
// Writes results/website.json (every check and whether it passed) and docs/screenshots/*.png.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BASE = process.argv[2] ?? "http://127.0.0.1:8000/";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PASSWORDS = "We store customer passwords. How should they be protected?";
const HTTPS_PORT = "What port number does HTTPS use by default?";
const NOT_COVERED = "SP 800-53 Rev 5.2.0 doesn't cover this question: none of the controls found for it answer it.";

const profile = mkdtempSync(join(tmpdir(), "control-finder-check-"));
const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--disable-extensions", "about:blank"], { stdio: "ignore" });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const checks = [];
const finish = (code) => {
  try { chrome.kill("SIGKILL"); } catch {}
  setTimeout(() => { rmSync(profile, { recursive: true, force: true }); process.exit(code); }, 400);
};
setTimeout(() => { console.log("TIMEOUT"); finish(1); }, 300000);

let targets = [];
for (let i = 0; i < 50 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter((t) => t.type === "page"); } catch {}
  await wait(200);
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
const problems = []; // console errors, exceptions, and Content Security Policy violations
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
  if (msg.method === "Runtime.exceptionThrown") problems.push(msg.params.exceptionDetails.text);
  if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") problems.push(msg.params.entry.text);
});
const send = (method, params = {}) => new Promise((resolve) => { const n = ++id; pending.set(n, resolve); ws.send(JSON.stringify({ id: n, method, params })); });
const js = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;
const until = async (expression, tries = 80) => { for (let i = 0; i < tries; i++) { if (await js(expression)) return true; await wait(250); } return false; };
const check = (name, ok, detail = "") => { checks.push({ name, passed: Boolean(ok) }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`); };
const click = (selector) => js(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (el) el.click(); return !!el; })()`);
const clickText = (selector, text) => js(`(() => { const el = [...document.querySelectorAll(${JSON.stringify(selector)})].find((e) => e.textContent.trim() === ${JSON.stringify(text)}); if (el) el.click(); return !!el; })()`);
const text = () => js("document.body.innerText");
const shot = async (name) => {
  // A tall window from the top of the page, so the picture shows the question and what came back.
  const { width, mobile } = await js("({ width: window.innerWidth, mobile: window.innerWidth < 500 })");
  await send("Emulation.setDeviceMetricsOverride", { width, height: mobile ? 844 : 1500, deviceScaleFactor: 1, mobile });
  await js("window.scrollTo(0, 0)");
  await wait(300);
  const { result } = await send("Page.captureScreenshot", { format: "png" });
  await send("Emulation.setDeviceMetricsOverride", { width, height: mobile ? 844 : 900, deviceScaleFactor: 1, mobile });
  mkdirSync(join(ROOT, "docs", "screenshots"), { recursive: true });
  writeFileSync(join(ROOT, "docs", "screenshots", `${name}.png`), Buffer.from(result.data, "base64"));
};
const view = async (width, height, mobile, scheme) => {
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme }] });
};
const load = async () => {
  await send("Page.navigate", { url: BASE });
  await until("document.querySelector('.status') && !document.querySelector('.status').textContent.startsWith('Checking')");
};

await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");

// ---- Desktop, light
await view(1280, 900, false, "light");
await load();
check("the page loads, titled Control Finder", (await js("document.title")) === "Control Finder");
check("the status line says everything is running", (await js("document.querySelector('.status').textContent")).startsWith("Ready: 1,014 controls"),
  await js("document.querySelector('.status').textContent"));
check("there's a skip link to the question", (await js("document.querySelector('a.skip')?.getAttribute('href')")) === "#question");
check("the form says the search choice is for Find controls, and answers always use Hybrid",
  (await text()).includes("This choice applies to Find controls. Answer with citations always uses Hybrid"));
await clickText("button.example", PASSWORDS);
check("an example fills in the question", (await js("document.querySelector('#question').value")) === PASSWORDS);
await click("form button[type=submit]");
const found = await until("!!document.querySelector('#results-heading')");
check("Find controls shows the 10 closest controls", found && (await js("document.querySelector('#results-heading').textContent")) === "The 10 closest controls");
check("focus moves to the results", (await js("document.activeElement?.id")) === "results-heading");
check("IA-5(1) Password-based Authentication is among them", (await text()).includes("IA-5(1) Password-based Authentication"));
check("the list says which choice found it", (await text()).includes("Searched by: Hybrid."));
await clickText("summary", "Read IA-5(1)");
const opened = await until("[...document.querySelectorAll('details.control-text')].some((d) => d.open && d.querySelector('.detail'))");
check("opening a control shows NIST's text, fetched from the API", opened && (await text()).includes("salted key derivation function"));
check("and what an assessor checks", (await text()).includes("What an assessor checks (SP 800-53A)"));
await clickText("summary", "Read IA-5(1)"); // closed again, so the picture shows the list
await shot("search-desktop-light");

await click("button.secondary");
const answered = await until("!!document.querySelector('#answer-heading')", 400);
check("Answer with citations gives an answer", answered);
check("the password question is covered and cites IA-5(1)", (await js("document.querySelector('.badge')?.textContent")) === "Covered"
  && (await text()).includes("Cited controls") && (await js("[...document.querySelectorAll('.result-head .label')].some((e) => e.textContent === 'IA-5(1)')")));
check("it says what the model saw", (await text()).includes("From a local model (qwen3:8b) that saw only the Hybrid search's top 10 controls"));
check("the controls it saw are listed by their NIST labels", /top 10 controls: IA-5\(1\)/.test(await text()) && !/ia-5\.1/.test(await text()));
await shot("answer-desktop-light");

await clickText("button.example", HTTPS_PORT);
await click("button.secondary");
await until("document.querySelector('.badge')?.textContent === 'Not covered'", 400);
check("a question the catalog doesn't cover gets Not covered", (await js("document.querySelector('.badge')?.textContent")) === "Not covered");
check("with the fixed sentence, not the model's own words", (await text()).includes(NOT_COVERED) && !(await text()).includes("443"));
check("and no citations", !(await text()).includes("Cited controls"));

await clickText("button.example", PASSWORDS);
await js("[...document.querySelectorAll('input[name=method]')].find((i) => i.value === 'keyword').click()");
await click("form button[type=submit]");
await until("document.querySelector('#results-heading') && document.body.innerText.includes('Searched by: Keywords only.')");
check("keyword search works too", (await text()).includes("Searched by: Keywords only.") && (await js("document.querySelectorAll('ol.results > li').length")) === 10);
const listed = () => js("[...document.querySelectorAll('ol.results > li .label')].map((e) => e.textContent).join(', ')");
const keywordList = await listed();
await js("[...document.querySelectorAll('input[name=method]')].find((i) => i.value === 'vector').click()");
await until("document.body.innerText.includes('You changed')", 20);
check("changing the choice after a search asks for a new one",
  (await text()).includes('You changed "Search by" to "Meaning only". Click Find controls to search again.'));
await click("form button[type=submit]");
await until("document.body.innerText.includes('Searched by: Meaning only.')");
check("and searching again gives that choice's own list", (await text()).includes("Searched by: Meaning only.") && (await listed()) !== keywordList
  && !(await text()).includes("You changed"), `keywords: ${keywordList} | meaning: ${await listed()}`);
const small = await js("[...document.querySelectorAll('button, summary, .choice')].filter((e) => e.offsetParent && e.getBoundingClientRect().height < 44).map((e) => e.textContent.trim().slice(0, 30))");
check("every button and toggle is at least 44 px tall", small.length === 0, small.join("; "));

// ---- Desktop, dark
await view(1280, 900, false, "dark");
await load();
check("dark mode follows the device", (await js("getComputedStyle(document.body).backgroundColor")) === "rgb(14, 18, 23)");
await clickText("button.example", PASSWORDS);
await click("button.secondary");
await until("!!document.querySelector('#answer-heading')", 400);
await shot("answer-desktop-dark");

// ---- Phone, light
await view(390, 844, true, "light");
await load();
await clickText("button.example", PASSWORDS);
await click("form button[type=submit]");
await until("!!document.querySelector('#results-heading')");
check("on a phone, nothing is wider than the screen", (await js("document.documentElement.scrollWidth <= window.innerWidth")),
  `${await js("document.documentElement.scrollWidth")} px in a ${await js("window.innerWidth")} px screen`);
await js("window.scrollTo(0, 0)");
await shot("search-phone-light");

check("no console errors or security-policy violations", problems.length === 0, problems.join(" | "));
const passed = checks.filter((c) => c.passed).length;
mkdirSync(join(ROOT, "results"), { recursive: true });
writeFileSync(join(ROOT, "results", "website.json"), JSON.stringify({ base: BASE, passed, total: checks.length, checks,
  screenshots: ["search-desktop-light", "answer-desktop-light", "answer-desktop-dark", "search-phone-light"] }, null, 1) + "\n");
console.log(`\n${passed} of ${checks.length} checks passed`);
finish(passed === checks.length ? 0 : 1);
