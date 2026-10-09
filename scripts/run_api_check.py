# ============================================================================
# Step 5: the API, run for real and checked against the published rankings.
#
# Run:   docker compose up -d --wait, and Ollama with nomic-embed-text
#        python scripts/run_api_check.py   (about 1 minute)
#   1. Starts the API (uvicorn, one worker) on a free local port, fresh for
#      each method, so no question has been asked before.
#   2. Sends one warm-up question (not in either answer key), which loads the
#      model and opens a database connection.
#   3. Asks all 203 questions over HTTP, twice: the first time each question is
#      new, the second time its embedding comes from the API's cache.
#   4. Compares every answer with the published lists: the top 10 for every
#      method, and hybrid's top 30, the reranker's candidates in step 4.
#   5. Writes results/api.json and docs/results-api.md.
# Every score uses the top 10, the same as steps 1 to 4.
# ============================================================================
import http.client
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder.answer_key import load_csf_questions, load_questions  # noqa: E402
from controlfinder.bm25 import BM25  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402
from controlfinder.fusion import rrf  # noqa: E402
from controlfinder.metrics import summarize  # noqa: E402
from controlfinder.search import DEPTH, RRF_K, VECTOR_WEIGHT  # noqa: E402

METHODS = ["keyword", "vector", "hybrid"]
ROUNDS = ["first time", "asked before"]
WARM_UP = "Which controls cover backups of system data?"
KEYS = {"csf2": ("CSF 2.0", load_csf_questions), "sp800_171": ("SP 800-171", load_questions)}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(port):
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "controlfinder.api:app", "--app-dir", "src",
                               "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            health = json.loads(conn.getresponse().read())
            conn.close()
            if health["status"] == "ok":
                return server, health
            raise SystemExit(f"the API started but isn't ready: {health}")
        except (ConnectionRefusedError, OSError):
            if server.poll() is not None:
                raise SystemExit("the API stopped:\n" + server.stderr.read().decode())
            time.sleep(0.2)
    raise SystemExit("the API didn't start within 60 s")


def ask(conn, question, method, k):
    path = "/search?" + urllib.parse.urlencode({"q": question, "method": method, "k": k})
    start = time.perf_counter()
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    elapsed = (time.perf_counter() - start) * 1000
    if response.status != 200:
        raise SystemExit(f"{response.status} for {question[:60]!r}: {body[:200]!r}")
    parts = {p.split(";dur=")[0]: float(p.split(";dur=")[1]) for p in response.getheader("Server-Timing").split(", ")}
    return [r["id"] for r in json.loads(body)["results"]], elapsed, parts


def ollama_version():
    conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=5)
    conn.request("GET", "/api/version")
    return json.loads(conn.getresponse().read())["version"]


def summary(values):
    ordered = sorted(values)
    return {"median": round(statistics.median(ordered), 2), "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 2)}


def main():
    started = time.time()
    controls = active_controls()
    keys = {name: load() for name, (_, load) in KEYS.items()}
    questions = [(name, q) for name, qs in keys.items() for q in qs]
    published = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())
    vector_lists = json.loads((ROOT / "results" / "vector_top100.json").read_text())[published["chosen"]["vector"]]
    rerank = json.loads((ROOT / "results" / "rerank.json").read_text())
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    keyword_lists = {name: {q.id: [controls[i].id for i, _ in bm25.search(q.text, DEPTH)] for q in qs} for name, qs in keys.items()}
    expected = {"keyword": lambda n, qid: keyword_lists[n][qid][:10],
                "vector": lambda n, qid: vector_lists[n][qid][:10],
                "hybrid": lambda n, qid: candidates[n][qid]}

    answers, same, timings, health = {}, {}, {}, None
    for method in METHODS:
        k = 30 if method == "hybrid" else 10
        port = free_port()
        server, health = start(port)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            ask(conn, WARM_UP, method, k)
            rounds = {}
            for round_name in ROUNDS:
                times, parts, got = [], {}, {}
                for name, q in questions:
                    ids, elapsed, server_parts = ask(conn, q.text, method, k)
                    got[(name, q.id)] = ids
                    times.append(elapsed)
                    for part, ms in server_parts.items():
                        parts.setdefault(part, []).append(ms)
                rounds[round_name] = {"request": summary(times), "server parts (median)": {p: round(statistics.median(v), 2) for p, v in parts.items()}}
                if round_name == ROUNDS[0]:
                    answers[method] = got
                elif got != answers[method]:
                    raise SystemExit(f"{method}: the second round's answers differ from the first's")
            timings[method] = rounds
        finally:
            server.terminate()
            server.wait(timeout=10)
        same[method] = {"top 10": sum(answers[method][(n, q.id)][:10] == expected[method](n, q.id)[:10] for n, q in questions)}
        if method == "hybrid":
            same[method]["top 30, the reranker's candidates"] = sum(answers[method][(n, q.id)] == expected[method](n, q.id) for n, q in questions)
        print(f"{method}: {same[method]} | {time.time() - started:.0f} s")

    def scores(method):
        return {n: summarize([(answers[method][(n, q.id)][:10], q.expected) for q in qs if q.split == "test"]) for n, qs in keys.items()}

    weighted = f"hybrid, vector weight {VECTOR_WEIGHT:g}"
    reproduced = {"keyword": all(scores("keyword")[n] == published["test"][n]["keyword (BM25)"] for n in keys),
                  "vector": all(scores("vector")[n] == published["test"][n]["vector"] for n in keys),
                  "hybrid": all(scores("hybrid")[n] == rerank["test"][n][weighted] for n in keys)}
    results = {
        "server": {"fastapi": version("fastapi"), "uvicorn": version("uvicorn"), "workers": 1,
                   "postgresql": health["postgresql"].split()[0], "pgvector": health["pgvector"],
                   "embedding_model": f"nomic-embed-text through Ollama {ollama_version()}", "settings": {"depth": DEPTH, "rrf_k": RRF_K, "vector_weight": VECTOR_WEIGHT}},
        "machine": f"{ {'Darwin': 'macOS'}.get(platform.system(), platform.system())} {platform.machine()}",
        "questions": len(questions),
        "same_as_published": same,
        "test_scores_reproduced": reproduced,
        "milliseconds": timings,
    }
    (ROOT / "results" / "api.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    doc = render_doc(results)
    (ROOT / "docs" / "results-api.md").write_text(doc, encoding="utf-8")
    print(doc)
    print(f"{time.time() - started:.0f} s")


def render_doc(r):
    """docs/results-api.md, from results/api.json alone (a test checks the committed page matches)."""
    n, s = r["questions"], r["server"]
    vector = {name: r["milliseconds"]["vector"][name]["server parts (median)"]["vector"] for name in ROUNDS}
    checks = []
    for method, counts in r["same_as_published"].items():
        checks.append(f"| {method} | " + "; ".join(f"{label}: {count} of {n}" for label, count in counts.items()) +
                      f" | {'yes' if r['test_scores_reproduced'][method] else 'no'} |")
    rows = []
    for method, rounds in r["milliseconds"].items():
        for round_name, m in rounds.items():
            parts = ", ".join(f"{p} {ms:.2f}" for p, ms in m["server parts (median)"].items())
            rows.append(f"| {method} | {round_name} | {m['request']['median']:.2f} | {m['request']['p95']:.2f} | {parts} |")
    return f"""# Results: the API (step 5)

Generated by `scripts/run_api_check.py`. The database behind it: [the vector database](results-database.md).

## Setup
- **The API:** FastAPI {s['fastapi']} on uvicorn {s['uvicorn']} (one worker), on this computer ({r['machine']}). PostgreSQL {s['postgresql']} with pgvector {s['pgvector']} in Docker, and {s['embedding_model']}.
- **Endpoints:** `GET /search` (`q`, `method` = hybrid, vector, or keyword, `k` up to 50), `GET /controls/{{id}}`, and `GET /health`. The interactive docs are at `/docs`.
- **Search:** keyword (BM25, in memory) and exact vector search in the database, each a top {s['settings']['depth']}, fused for hybrid with reciprocal rank fusion (k = {s['settings']['rrf_k']}) and the vector list weighted {s['settings']['vector_weight']:g} times, as chosen in step 4.
- **The run:** a fresh API for each method, one warm-up question, then all {n} questions twice over HTTP: first when each question is new, then when it has been asked before (its embedding comes from the API's cache).

## It returns the published rankings
| Method | Same answers as the published lists | Published test scores come back |
|---|---|---|
{chr(10).join(checks)}

The questions were embedded by the running model, not read from the committed embeddings, so this checks the whole path: the request, the embedding and its query prefix, the database search, and the fusion.

## Speed
Milliseconds per request, measured from the client, and the server's own timing of each part (`Server-Timing` header, medians).

| Method | Question | Median (ms) | 95th percentile (ms) | Server parts, median (ms) |
|---|---|---|---|---|
{chr(10).join(rows)}

## Reading these numbers
- **embed:** turning the question into a vector with the local model; skipped when the question was asked before. **vector:** exact search in the database. **keyword:** BM25 in memory. **fuse:** reciprocal rank fusion.
- In the first round, the other parts are slower too: the database search took {vector[ROUNDS[0]]:.2f} ms, against {vector[ROUNDS[1]]:.2f} ms when the question was asked before. It runs right after the model, on the same computer; the cause wasn't investigated.
- Timings depend on the computer and include the trip from the client and back, so CI doesn't check them. CI does check the answers, with the committed embeddings standing in for the model.
"""


if __name__ == "__main__":
    main()
