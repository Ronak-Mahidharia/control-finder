import { labelOf, type Fetch } from "../lib/api";
import type { AnswerResponse } from "../lib/types";
import { ControlText } from "./ControlText";

export function AnswerCard({ data, seconds, fetchFn }: { data: AnswerResponse; seconds: number | null; fetchFn: Fetch }) {
  return (
    <section aria-labelledby="answer-heading" className="card">
      <h2 id="answer-heading" tabIndex={-1}>
        Answer
      </h2>
      <p>
        <span className={data.covered ? "badge covered" : "badge not-covered"}>{data.covered ? "Covered" : "Not covered"}</span>
        {seconds !== null && <span className="muted small"> in {seconds.toFixed(1)} s</span>}
      </p>
      <p className="answer-text">{data.answer}</p>
      {data.citations.length > 0 && (
        <>
          <h3>Cited controls</h3>
          <ul className="results">
            {data.citations.map((c) => (
              <li key={c.id}>
                <p className="result-head">
                  <span className="label">{c.label}</span> <span className="title">{c.title}</span>
                </p>
                <ControlText id={c.id} label={c.label} fetchFn={fetchFn} />
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="muted small">
        From a local model ({data.model}) that saw only the search's top {data.shown.length} controls: {data.shown.map(labelOf).join(", ")}. Its
        citations can only name those. Check a cited control's text before relying on it.
      </p>
    </section>
  );
}
