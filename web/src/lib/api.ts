// Calls to the Control Finder API that serves this page (same origin, so no other server is involved).
import type { AnswerResponse, ControlDetail, Health, Method, SearchResponse } from "./types";

export type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export const MAX_QUESTION = 2000; // the API's limit, in characters
export const START_HINT = "Start everything with scripts/start_local.sh, then reload this page.";

// fetch has to be called on the window, so it's wrapped instead of passed around on its own.
export const browserFetch: Fetch = (input, init) => fetch(input, init);

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

function detailOf(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI's checks on the question, for example "String should have at most 2000 characters"
    const messages = detail.map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : "")).filter(Boolean);
    return messages.length ? `The question wasn't accepted: ${messages.join("; ")}.` : null;
  }
  return null;
}

async function call<T>(path: string, fetchFn: Fetch, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetchFn(path, { signal, headers: { Accept: "application/json" } });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError(`The Control Finder API isn't reachable. ${START_HINT}`, 0);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(detailOf(body) ?? `The API answered ${response.status}.`, response.status);
  return body as T;
}

export const searchPath = (question: string, method: Method, k: number) =>
  `/search?${new URLSearchParams({ q: question, method, k: String(k) })}`;
export const answerPath = (question: string) => `/answer?${new URLSearchParams({ q: question })}`;
export const controlPath = (id: string) => `/controls/${encodeURIComponent(id)}`;

export const search = (question: string, method: Method, k = 10, fetchFn: Fetch = browserFetch, signal?: AbortSignal) =>
  call<SearchResponse>(searchPath(question, method, k), fetchFn, signal);
export const answer = (question: string, fetchFn: Fetch = browserFetch, signal?: AbortSignal) =>
  call<AnswerResponse>(answerPath(question), fetchFn, signal);
export const control = (id: string, fetchFn: Fetch = browserFetch, signal?: AbortSignal) =>
  call<ControlDetail>(controlPath(id), fetchFn, signal);
export const health = (fetchFn: Fetch = browserFetch, signal?: AbortSignal) => call<Health>("/health", fetchFn, signal);

// What's running, in plain words, from /health.
export function statusLine(h: Health | "unreachable" | null): { ok: boolean; text: string } {
  if (h === null) return { ok: true, text: "Checking what's running…" };
  if (h === "unreachable") return { ok: false, text: `The API isn't reachable. ${START_HINT}` };
  const missing = [h.database === "unreachable" ? "the database" : "", h.embedding_model === "unreachable" ? "the embedding model (Ollama)" : ""].filter(Boolean);
  if (h.status === "ok" && !missing.length) return { ok: true, text: `Ready: ${h.controls.toLocaleString("en-US")} controls and enhancements, PostgreSQL with pgvector ${h.pgvector}, and the local embedding model.` };
  const why = missing.length ? `: ${missing.join(" and ")} can't be reached` : ""; // "degraded" is enough to warn, even without a reason
  return { ok: false, text: `Only keyword search works right now${why}. ${START_HINT}` };
}

// An OSCAL id as NIST prints it: "ia-5.1" becomes "IA-5(1)", and "ac-2" becomes "AC-2".
export function labelOf(id: string): string {
  const m = /^([a-z]{2})-(\d+)(?:\.(\d+))?$/.exec(id);
  if (!m) return id;
  return `${m[1].toUpperCase()}-${m[2]}${m[3] ? `(${m[3]})` : ""}`;
}
