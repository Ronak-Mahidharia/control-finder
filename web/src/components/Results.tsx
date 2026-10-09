import type { Fetch } from "../lib/api";
import type { Found, SearchResponse } from "../lib/types";
import { ControlText } from "./ControlText";

const METHOD_NOTES = {
  hybrid: "Hybrid search: keyword and meaning-based results fused, with the meaning-based list counted 3 times (the setting chosen on NIST's CSF 2.0 questions).",
  vector: "Search by meaning, with the local embedding model (nomic-embed-text).",
  keyword: "Keyword search (BM25). It works even when the database or the embedding model isn't running.",
};

function ranks(r: Found): string {
  const parts = [r.keyword_rank !== null ? `keyword rank ${r.keyword_rank}` : "", r.vector_rank !== null ? `meaning rank ${r.vector_rank}` : ""];
  return parts.filter(Boolean).join(", ");
}

export function Results({ data, fetchFn }: { data: SearchResponse; fetchFn: Fetch }) {
  const n = data.results.length;
  return (
    <section aria-labelledby="results-heading" className="card">
      <h2 id="results-heading" tabIndex={-1}>
        {n === 0 ? "No controls found" : `The ${n} closest control${n === 1 ? "" : "s"}`}
      </h2>
      <p className="muted">{METHOD_NOTES[data.method]}</p>
      {n === 0 && <p>No control shares the question's words. Try hybrid search, which also searches by meaning.</p>}
      <ol className="results">
        {data.results.map((r) => (
          <li key={r.id}>
            <p className="result-head">
              <span className="label">{r.label}</span> <span className="title">{r.title}</span>
            </p>
            <p className="muted small">
              {r.family}
              {ranks(r) && ` · ${ranks(r)}`}
            </p>
            <ControlText id={r.id} label={r.label} fetchFn={fetchFn} />
          </li>
        ))}
      </ol>
      <p className="muted small">These are candidates ranked by fit, not a compliance decision: which controls a system needs depends on its baseline and risk assessment.</p>
    </section>
  );
}
