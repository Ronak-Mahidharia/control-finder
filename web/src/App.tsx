import { useEffect, useRef, useState } from "react";
import { AnswerCard } from "./components/AnswerCard";
import { METHOD_NAMES, Results } from "./components/Results";
import { answer, browserFetch, health, MAX_QUESTION, search, statusLine, type Fetch } from "./lib/api";
import type { AnswerResponse, Health, Method, SearchResponse } from "./lib/types";

// Two the catalog covers, one it partly covers, and one it doesn't (from the hand-made question sets).
export const EXAMPLES = [
  "We store customer passwords. How should they be protected?",
  "How often should we back up our systems' data?",
  "Do we need a firewall between our network and the internet?",
  "What port number does HTTPS use by default?",
];

export const METHODS: { value: Method; label: string }[] = [
  { value: "hybrid", label: "Hybrid (keywords and meaning, the best measured)" },
  { value: "vector", label: "Meaning only" },
  { value: "keyword", label: "Keywords only (works without the database and Ollama)" },
];

export type View =
  | { kind: "idle" }
  | { kind: "busy"; what: "search" | "answer" }
  | { kind: "results"; data: SearchResponse }
  | { kind: "answer"; data: AnswerResponse; seconds: number | null }
  | { kind: "error"; message: string; field?: "question" };

// Where keyboard and screen-reader focus goes when something comes back.
function focusTarget(view: View): string | null {
  if (view.kind === "results") return "results-heading";
  if (view.kind === "answer") return "answer-heading";
  if (view.kind === "error") return view.field ?? "error-message";
  return null;
}

interface PageProps {
  question: string;
  method: Method;
  view: View;
  status: Health | "unreachable" | null;
  fetchFn: Fetch;
  onQuestion: (q: string) => void;
  onMethod: (m: Method) => void;
  onRun: (what: "search" | "answer") => void;
}

// The whole page, from its state. It has no state of its own, so tests can render every case.
export function Page({ question, method, view, status, fetchFn, onQuestion, onMethod, onRun }: PageProps) {
  const s = statusLine(status);
  const busy = view.kind === "busy";
  // The list on the page came from another choice than the one now picked.
  const changedTo = view.kind === "results" && view.data.method !== method ? method : null;
  return (
    <>
      <a className="skip" href="#question">
        Skip to the question
      </a>
      <header className="site-header">
        <h1>Control Finder</h1>
        <p className="tagline">Find the NIST SP 800-53 controls that answer a security question, each one cited.</p>
        <p className={s.ok ? "status" : "status warn"} role="status">
          {s.text}
        </p>
      </header>
      <main>
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            onRun("search");
          }}
        >
          <label htmlFor="question" className="field-label">
            Your question
          </label>
          <textarea
            id="question"
            value={question}
            maxLength={MAX_QUESTION}
            rows={3}
            placeholder="For example: How often should we back up our systems' data?"
            onChange={(e) => onQuestion(e.target.value)}
          />
          <p className="muted small">
            {question.length.toLocaleString("en-US")} of {MAX_QUESTION.toLocaleString("en-US")} characters
          </p>
          <fieldset aria-describedby="method-note">
            <legend className="field-label">Search by</legend>
            {METHODS.map((m) => (
              <label key={m.value} className="choice">
                <input type="radio" name="method" value={m.value} checked={method === m.value} onChange={() => onMethod(m.value)} />
                {m.label}
              </label>
            ))}
            <p id="method-note" className="muted small">
              This choice applies to Find controls. Answer with citations always uses Hybrid, the setup its answers were tested with.
            </p>
          </fieldset>
          <div className="actions">
            <button type="submit" disabled={busy}>
              Find controls
            </button>
            <button type="button" className="secondary" disabled={busy} onClick={() => onRun("answer")}>
              Answer with citations
            </button>
          </div>
          <div className="examples">
            <p className="field-label">Or try one</p>
            <ul>
              {EXAMPLES.map((q) => (
                <li key={q}>
                  <button type="button" className="example" disabled={busy} onClick={() => onQuestion(q)}>
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </form>
        <div aria-live="polite" className="busy">
          {view.kind === "busy" &&
            (view.what === "search" ? "Searching…" : "Asking the local model. An answer takes a few seconds, longer the first time.")}
          {changedTo && (
            <span className="notice">
              You changed "Search by" to "{METHOD_NAMES[changedTo]}". Click Find controls to search again.
            </span>
          )}
        </div>
        {view.kind === "error" && (
          <p role="alert" id="error-message" tabIndex={-1} className="card error">
            {view.message}
          </p>
        )}
        {view.kind === "results" && <Results data={view.data} fetchFn={fetchFn} />}
        {view.kind === "answer" && <AnswerCard data={view.data} seconds={view.seconds} fetchFn={fetchFn} />}
      </main>
      <footer className="site-footer">
        <p>
          Not affiliated with or endorsed by NIST. A reference tool, not legal or compliance advice. The data is NIST SP 800-53 Rev
          5.2.0 with SP 800-53A, in the public domain.
        </p>
        <p>
          Everything runs on this computer: the API, the database, and the local models. <a href="/docs">The API's own page</a>.
        </p>
      </footer>
    </>
  );
}

export function App({ fetchFn = browserFetch }: { fetchFn?: Fetch }) {
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<Method>("hybrid");
  const [view, setView] = useState<View>({ kind: "idle" });
  const [status, setStatus] = useState<Health | "unreachable" | null>(null);
  const running = useRef<AbortController | null>(null);
  const startedFrom = useRef<Element | null>(null);

  useEffect(() => {
    health(fetchFn).then(setStatus, () => setStatus("unreachable"));
  }, [fetchFn]);

  useEffect(() => {
    // Move focus to what came back, unless the person has moved on while waiting (say, back into the
    // question box): then it stays where they put it.
    const target = focusTarget(view);
    const here = document.activeElement;
    if (target && (!here || here === document.body || here === startedFrom.current)) document.getElementById(target)?.focus();
  }, [view]);

  const run = async (what: "search" | "answer") => {
    startedFrom.current = document.activeElement;
    const q = question.trim();
    if (!q) {
      setView({ kind: "error", message: "Type a question first, or pick an example.", field: "question" });
      return;
    }
    running.current?.abort();
    const controller = new AbortController();
    running.current = controller;
    setView({ kind: "busy", what });
    const started = performance.now();
    try {
      if (what === "search") {
        setView({ kind: "results", data: await search(q, method, 10, fetchFn, controller.signal) });
      } else {
        const data = await answer(q, fetchFn, controller.signal);
        setView({ kind: "answer", data, seconds: (performance.now() - started) / 1000 });
      }
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") return;
      setView({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <Page
      question={question}
      method={method}
      view={view}
      status={status}
      fetchFn={fetchFn}
      onQuestion={setQuestion}
      onMethod={setMethod}
      onRun={run}
    />
  );
}
