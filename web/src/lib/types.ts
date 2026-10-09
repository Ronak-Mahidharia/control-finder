// The API's answers, as src/controlfinder/api.py defines them.
export type Method = "hybrid" | "vector" | "keyword";

export interface Found {
  rank: number;
  id: string;
  label: string;
  title: string;
  family: string;
  keyword_rank: number | null;
  vector_rank: number | null;
}

export interface SearchResponse {
  query: string;
  method: Method;
  vector_weight: number | null;
  results: Found[];
}

export interface Citation {
  id: string;
  label: string;
  title: string;
}

export interface AnswerResponse {
  query: string;
  covered: boolean;
  answer: string;
  citations: Citation[];
  shown: string[];
  model: string;
}

export interface ControlDetail {
  id: string;
  label: string;
  title: string;
  family: string;
  parent: string | null;
  statement: string;
  guidance: string;
  assessment_objectives: string[];
}

export interface Health {
  status: "ok" | "degraded";
  controls: number;
  database: "ready" | "loaded" | "unreachable";
  postgresql: string | null;
  pgvector: string | null;
  embedding_model: "reachable" | "unreachable";
}
