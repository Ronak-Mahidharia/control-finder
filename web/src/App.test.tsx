// The page, rendered to HTML without a browser, in each state it can be in.
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CONTENT_SECURITY_POLICY } from "../vite.config";
import { EXAMPLES, Page, type View } from "./App";
import { Detail } from "./components/ControlText";
import type { Fetch } from "./lib/api";
import type { AnswerResponse, SearchResponse } from "./lib/types";

const noFetch: Fetch = async () => {
  throw new Error("no network in these tests");
};
const render = (view: View, question = "We store customer passwords. How should they be protected?") =>
  renderToStaticMarkup(
    <Page question={question} method="hybrid" view={view} status={null} fetchFn={noFetch} onQuestion={() => {}} onMethod={() => {}} onRun={() => {}} />,
  );

const results: SearchResponse = {
  query: "q",
  method: "hybrid",
  vector_weight: 3,
  results: [
    { rank: 1, id: "ia-5.18", label: "IA-5(18)", title: "Password Managers", family: "Identification and Authentication", keyword_rank: 2, vector_rank: 2 },
    { rank: 2, id: "ia-5.1", label: "IA-5(1)", title: "Password-based Authentication", family: "Identification and Authentication", keyword_rank: 1, vector_rank: null },
  ],
};
const covered: AnswerResponse = {
  query: "q",
  covered: true,
  answer: "Store them with an approved salted key derivation function.",
  citations: [{ id: "ia-5.1", label: "IA-5(1)", title: "Password-based Authentication" }],
  shown: ["ia-5.18", "ia-5.1"], // the API sends ids
  model: "qwen3:8b",
};

describe("the page", () => {
  it("starts with the question, the search choices, and the examples", () => {
    const html = render({ kind: "idle" }, "");
    expect(html).toContain('<label for="question" class="field-label">Your question</label>');
    expect(html).toContain('maxLength="2000"');
    expect(html.match(/type="radio"/g)).toHaveLength(3);
    for (const q of EXAMPLES) expect(html).toContain(q.replace(/'/g, "&#x27;"));
    expect(html).toContain("Not affiliated with or endorsed by NIST");
    expect(html).toContain("0 of 2,000 characters");
  });

  it("shows the search results with each list's rank", () => {
    const html = render({ kind: "results", data: results });
    expect(html).toContain("The 2 closest controls");
    expect(html).toContain('<span class="label">IA-5(1)</span> <span class="title">Password-based Authentication</span>');
    expect(html).toContain("keyword rank 2, meaning rank 2");
    expect(html).toContain("Identification and Authentication · keyword rank 1<");
    expect(html).toContain("not a compliance decision");
    expect(html.match(/<summary>Read IA-5/g)).toHaveLength(2);
  });

  it("says when keyword search finds nothing", () => {
    const html = render({ kind: "results", data: { ...results, method: "keyword", results: [] } });
    expect(html).toContain("No controls found");
    expect(html).toContain("Try hybrid search");
  });

  it("shows a covered answer with its citations and what the model saw", () => {
    const html = render({ kind: "answer", data: covered, seconds: 4.33 });
    expect(html).toContain('<span class="badge covered">Covered</span>');
    expect(html).toContain("in 4.3 s");
    expect(html).toContain("Cited controls");
    expect(html).toContain("From a local model (qwen3:8b) that saw only the search&#x27;s top 2 controls: IA-5(18), IA-5(1)."); // as NIST prints them
  });

  it("shows a not-covered answer without citations", () => {
    const fixed = "SP 800-53 Rev 5.2.0 doesn't cover this question: none of the controls found for it answer it.";
    const html = render({ kind: "answer", data: { ...covered, covered: false, answer: fixed, citations: [] }, seconds: null });
    expect(html).toContain('<span class="badge not-covered">Not covered</span>');
    expect(html).toContain(fixed.replace(/'/g, "&#x27;"));
    expect(html).not.toContain("Cited controls");
  });

  it("tells the person while an answer is coming, and blocks a second request", () => {
    const html = render({ kind: "busy", what: "answer" });
    expect(html).toContain("Asking the local model");
    expect(html.match(/<button[^>]*disabled=""/g)?.length).toBeGreaterThanOrEqual(2);
  });

  it("shows errors as alerts", () => {
    const html = render({ kind: "error", message: "the database isn't reachable" });
    expect(html).toContain('role="alert"');
    expect(html).toContain("the database isn&#x27;t reachable");
  });
});

describe("a control's text", () => {
  it("shows the statement, what an assessor checks, and NIST's guidance", () => {
    const html = renderToStaticMarkup(
      <Detail d={{ id: "ia-5.1", label: "IA-5(1)", title: "Password-based Authentication", family: "IA", parent: "ia-5", statement: "Store passwords safely.", guidance: "Use a salted KDF.", assessment_objectives: ["Passwords are stored safely."] }} />,
    );
    expect(html).toContain("<p>Store passwords safely.</p>");
    expect(html).toContain("<li>Passwords are stored safely.</li>");
    expect(html).toContain("<summary>NIST&#x27;s guidance</summary>");
  });
});

describe("the built page's security policy", () => {
  it("lets the page call only its own API, with no inline scripts", () => {
    const directives = Object.fromEntries(CONTENT_SECURITY_POLICY.split("; ").map((d) => [d.split(" ")[0], d.split(" ").slice(1).join(" ")]));
    expect(directives).toEqual({
      "default-src": "'self'", // scripts and styles too: only the site's own files, never inline
      "connect-src": "'self'", // the API that serves the page, and nothing else
      "img-src": "'self' data:",
      "object-src": "'none'",
      "base-uri": "'none'",
      "form-action": "'none'",
    });
  });
});
