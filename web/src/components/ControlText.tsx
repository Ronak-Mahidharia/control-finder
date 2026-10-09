import { useState } from "react";
import { control, type Fetch } from "../lib/api";
import type { ControlDetail } from "../lib/types";

// A control's own text, fetched the first time it's opened.
export function ControlText({ id, label, fetchFn, initial = null }: { id: string; label: string; fetchFn: Fetch; initial?: ControlDetail | null }) {
  const [detail, setDetail] = useState<ControlDetail | null>(initial);
  const [error, setError] = useState<string | null>(null);
  const load = (open: boolean) => {
    if (!open || detail) return;
    setError(null);
    control(id, fetchFn).then(setDetail, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  };
  return (
    <details className="control-text" onToggle={(e) => load(e.currentTarget.open)}>
      <summary>Read {label}</summary>
      {error ? <p role="alert" className="error">{error}</p> : detail ? <Detail d={detail} /> : <p className="muted">Loading…</p>}
    </details>
  );
}

export function Detail({ d }: { d: ControlDetail }) {
  return (
    <div className="detail">
      <h4>Statement</h4>
      <p>{d.statement}</p>
      {d.assessment_objectives.length > 0 && (
        <>
          <h4>What an assessor checks (SP 800-53A)</h4>
          <ul>
            {d.assessment_objectives.map((objective, i) => (
              <li key={i}>{objective}</li>
            ))}
          </ul>
        </>
      )}
      {d.guidance && (
        <details>
          <summary>NIST's guidance</summary>
          <p>{d.guidance}</p>
        </details>
      )}
    </div>
  );
}
