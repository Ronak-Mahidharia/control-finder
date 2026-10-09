// The API client with a stand-in fetch: the addresses it asks for, and how it reads answers and errors.
import { describe, expect, it } from "vitest";
import { answerPath, ApiError, controlPath, labelOf, search, searchPath, statusLine, type Fetch } from "./api";
import type { Health } from "./types";

const json = (status: number, body: unknown): Response => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("addresses", () => {
  it("encodes the question", () => {
    expect(searchPath("Who may enter the server room? A & B", "hybrid", 10)).toBe("/search?q=Who+may+enter+the+server+room%3F+A+%26+B&method=hybrid&k=10");
    expect(answerPath("café #1")).toBe("/answer?q=caf%C3%A9+%231");
    expect(controlPath("ia-5.1")).toBe("/controls/ia-5.1");
  });
});

describe("calls", () => {
  it("returns the API's answer", async () => {
    const asked: string[] = [];
    const fetchFn: Fetch = async (input) => {
      asked.push(input);
      return json(200, { query: "q", method: "keyword", vector_weight: null, results: [] });
    };
    const out = await search("q", "keyword", 5, fetchFn);
    expect(asked).toEqual(["/search?q=q&method=keyword&k=5"]);
    expect(out.method).toBe("keyword");
  });

  it("passes on the API's reason when it can't answer", async () => {
    const down: Fetch = async () => json(503, { detail: "the database isn't reachable; keyword search still works (method=keyword)" });
    await expect(search("q", "hybrid", 10, down)).rejects.toThrow(new ApiError("the database isn't reachable; keyword search still works (method=keyword)", 503));
  });

  it("explains a refused question", async () => {
    const refused: Fetch = async () => json(422, { detail: [{ msg: "String should have at most 2000 characters" }] });
    await expect(search("q", "hybrid", 10, refused)).rejects.toThrow("The question wasn't accepted: String should have at most 2000 characters.");
  });

  it("says when the API isn't running", async () => {
    const offline: Fetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    await expect(search("q", "hybrid", 10, offline)).rejects.toThrow(/isn't reachable\. Start everything with scripts\/start_local\.sh/);
  });

  it("lets a cancelled request stay cancelled", async () => {
    const cancelled: Fetch = async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    };
    await expect(search("q", "hybrid", 10, cancelled)).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("status line", () => {
  const ready: Health = { status: "ok", controls: 1014, database: "ready", postgresql: "18.6", pgvector: "0.8.7", embedding_model: "reachable" };
  it("says what's running", () => {
    expect(statusLine(null)).toEqual({ ok: true, text: "Checking what's running…" });
    expect(statusLine(ready).text).toBe("Ready: 1,014 controls and enhancements, PostgreSQL with pgvector 0.8.7, and the local embedding model.");
    expect(statusLine({ ...ready, status: "degraded", database: "unreachable", embedding_model: "unreachable" })).toEqual({
      ok: false,
      text: "Only keyword search works right now: the database and the embedding model (Ollama) can't be reached. Start everything with scripts/start_local.sh, then reload this page.",
    });
    expect(statusLine("unreachable").ok).toBe(false);
  });
});

describe("labels", () => {
  it("prints ids the way NIST does", () => {
    expect(labelOf("ia-5.1")).toBe("IA-5(1)");
    expect(labelOf("ac-2")).toBe("AC-2");
    expect(labelOf("sc-7.27")).toBe("SC-7(27)");
    expect(labelOf("not-an-id")).toBe("not-an-id");
  });
});
