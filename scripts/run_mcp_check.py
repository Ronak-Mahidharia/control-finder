# ============================================================================
# Step 7: the MCP server, run the way an AI assistant runs it (over stdio,
# with the real models) and checked against the published rankings.
#
# Run:   Ollama with nomic-embed-text and qwen3:8b
#        python scripts/run_mcp_check.py   (about 20 seconds)
#   1. Starts the server as its own process, the way Claude Desktop or Claude
#      Code would, and times the connection and the list of tools.
#   2. Asks search_controls all 203 questions (hybrid, top 20) and compares
#      each answer with the published lists: the top 20 of the step 4
#      candidates, which the reranker scored.
#   3. Asks answer_question three questions: one plain-English, one the
#      catalog doesn't cover, and one from NIST's key.
#   4. Writes results/mcp.json and docs/results-mcp.md.
# ============================================================================
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder.answer_key import load_csf_questions, load_out_of_scope, load_plain_questions, load_questions  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402

K = 20
ASKED = [("plain_english", "plain-01"), ("out_of_scope", "oos-06"), ("sp800_171", "03.05.07")]
SETS = {"csf2": load_csf_questions, "sp800_171": load_questions, "plain_english": load_plain_questions,
        "out_of_scope": load_out_of_scope}


def summary(ms):
    ordered = sorted(ms)
    return {"median": round(statistics.median(ordered), 1), "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 1)}


async def check():
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]
    questions = [(key, q) for key in ("csf2", "sp800_171") for q in SETS[key]()]
    params = StdioServerParameters(command=sys.executable, args=["-m", "controlfinder.mcp_server"], cwd=str(ROOT),
                                   env={"PYTHONPATH": str(ROOT / "src")})
    out = {}
    start = time.perf_counter()
    async with Client(params) as client:
        out["connect_ms"] = round((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        tools = (await client.list_tools()).tools
        out["list_tools_ms"] = round((time.perf_counter() - start) * 1000)
        out["tools"] = sorted(t.name for t in tools)
        times, same, first = [], 0, None
        for key, q in questions:
            start = time.perf_counter()
            result = await client.call_tool("search_controls", {"question": q.text, "k": K})
            elapsed = (time.perf_counter() - start) * 1000
            if result.is_error:
                raise SystemExit(f"search_controls failed for {q.id}: {result.content[0].text}")
            if first is None:
                first = round(elapsed)  # loads the catalog, the keyword index, and the embeddings
            else:
                times.append(elapsed)
            same += [r["id"] for r in result.structured_content["results"]] == candidates[key][q.id][:K]
        out["search"] = {"questions": len(questions), "same_top_20_as_published": same, "first_call_ms": first,
                         "later_calls_ms": summary(times)}
        by_id = {(key, q.id): q for key, load in SETS.items() for q in load()}
        labels = {c.id: c.label for c in active_controls()}
        answers = []
        for key, qid in ASKED:
            q = by_id[(key, qid)]
            start = time.perf_counter()
            result = await client.call_tool("answer_question", {"question": q.text})
            seconds = round(time.perf_counter() - start, 2)
            if result.is_error:
                raise SystemExit(f"answer_question failed for {qid}: {result.content[0].text}")
            a = result.structured_content
            expected = sorted(labels[c] for c in getattr(q, "expected", ()))  # NIST's links, or the anchors
            answers.append({"set": key, "id": qid, "question": q.text, "expected": expected, "covered": a["covered"],
                            "answer": a["answer"], "citations": [c["label"] for c in a["citations"]], "seconds": seconds})
        out["answers"] = answers
    return out


def main():
    started = time.time()
    out = anyio.run(check)
    results = {"server": {"mcp": version("mcp"), "transport": "stdio", "searches_in": "memory (step 5: same rankings as the database)"},
               "machine": f"{ {'Darwin': 'macOS'}.get(platform.system(), platform.system())} {platform.machine()}", **out}
    (ROOT / "results" / "mcp.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    doc = render_doc(results)
    (ROOT / "docs" / "results-mcp.md").write_text(doc, encoding="utf-8")
    print(doc)
    print(f"{time.time() - started:.0f} s")


def render_doc(r):
    """docs/results-mcp.md, from results/mcp.json alone (a test checks the committed page matches)."""
    s = r["search"]
    asked = "\n".join(f"- **{a['id']}**: \"{a['question']}\" → {'covered' if a['covered'] else 'not covered'}"
                      + (f", citing {', '.join(a['citations'])}" if a["citations"] else "")
                      + f" ({a['seconds']:.2f} s). The answer: \"{a['answer']}\""
                      + (f" Expected: {', '.join(a['expected'])}" + (f"; {len(a['citations'])} controls were cited." if len(a["citations"]) > 2 else ".")
                         if a["expected"] else "") for a in r["answers"])
    return f"""# Results: the MCP server (step 7)

Generated by `scripts/run_mcp_check.py`. The answers it gives are measured in [cited answers](results-answers.md).

## Setup
- **The server:** `python -m controlfinder.mcp_server`, on the official MCP Python SDK (mcp {r['server']['mcp']}), over {r['server']['transport']}, started as its own process the way Claude Desktop or Claude Code start it, on this computer ({r['machine']}).
- **Tools** ({len(r['tools'])}, all read-only): {', '.join(f'`{t}`' for t in r['tools'])}.
- **Search** runs in {r['server']['searches_in']}, so the server needs Ollama but no database.

## It returns the published rankings
- `search_controls` gave the published top {K} for {s['same_top_20_as_published']} of {s['questions']} questions, with the model embedding each question live.

## Speed
- **Connecting** (starting the server and the MCP handshake): {r['connect_ms']} ms. **Listing the tools:** {r['list_tools_ms']} ms.
- **The first search:** {s['first_call_ms']} ms, which loads the catalog, the keyword index, and the embeddings. **Later searches:** {s['later_calls_ms']['median']:.1f} ms median, {s['later_calls_ms']['p95']:.1f} ms at the 95th percentile.

## Answers (live, word for word)
{asked}

## Reading these numbers
- **Expected** is what NIST links the question to, or the plain-English question's anchors. Citing more controls than that is the main weakness measured in [cited answers](results-answers.md).
- Times depend on the computer, so CI doesn't check them. CI does check the tools, their limits, and every question's search results through an MCP client, with the committed embeddings standing in for the model.
"""


if __name__ == "__main__":
    main()
