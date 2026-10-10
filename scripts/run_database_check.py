# ============================================================================
# Step 5: the controls in PostgreSQL with pgvector, checked against the
# in-memory vector search of steps 3 and 4.
#
# Run:   docker compose up -d --wait   (the database in docker-compose.yml)
#        python scripts/run_database_check.py   (about 1 minute)
#   1. Loads the committed embeddings (embeddings/) into a fresh table, in a
#      database of its own on that server (db.CHECK_DATABASE), so the site's
#      table isn't touched.
#   2. Exact search: every question's top 100, compared with the committed
#      in-memory lists (results/vector_top100.json), then scored on the test
#      questions alone and inside the hybrid searches of steps 3 and 4.
#   3. HNSW index: rows returned, how much of the exact top 10 and top 100 it
#      finds, and what changes downstream, at pgvector's default ef_search (40)
#      and at 100 and 200. The index places each control at a random level, so
#      it's built 10 times.
#   4. Speed: the time per question for each search, measured from Python.
#   5. Writes results/database.json and docs/results-database.md.
# Every score uses the top 10, the same as steps 1 to 4.
# ============================================================================
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder import db  # noqa: E402
from controlfinder.answer_key import load_csf_questions, load_questions  # noqa: E402
from controlfinder.bm25 import BM25  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402
from controlfinder.fusion import rrf  # noqa: E402
from controlfinder.metrics import summarize  # noqa: E402
from controlfinder.vector import rank_gap, search  # noqa: E402

DEPTH = 100
VECTOR_TEXT = "title + statement + guidance"  # chosen in step 3
EF_SEARCH = [None, 100, 200]  # None: pgvector's default, 40
BUILDS = 10
TIMED_ROUNDS = 10
KEYS = {"csf2": ("CSF 2.0", load_csf_questions), "sp800_171": ("SP 800-171", load_questions)}
SHOWN = {"mrr": "MRR", "hit@1": "first result right", "hit@10": "a right control in the top 10"}
IMAGE = "pgvector/pgvector:0.8.7-pg18@sha256:2358fcba361ed2233a5ed81b5fe4ca779ccb304120ce531a3bf51c0ed7e2bc11"


def ef_name(ef):
    return "ef_search 40 (default)" if ef is None else f"ef_search {ef}"


def load_embeddings(controls, keys):
    manifest = json.loads((ROOT / "embeddings" / "manifest.json").read_text())
    assert manifest["controls"]["ids"] == [c.id for c in controls], "the embeddings don't match the catalog"
    docs = np.load(ROOT / "embeddings" / manifest["controls"]["file"])
    questions = {}
    for key, qs in keys.items():
        assert manifest["questions"][key]["ids"] == [q.id for q in qs], f"the embeddings don't match the {key} questions"
        questions[key] = dict(zip(manifest["questions"][key]["ids"], np.load(ROOT / "embeddings" / manifest["questions"][key]["file"])))
    return docs, questions


def score(rankings, questions):
    return summarize([(rankings[q.id][:10], q.expected) for q in questions if q.split == "test"])


def overlap(got, want, k):
    return len(set(got[:k]) & set(want[:k])) / k


def spread(values):
    return {"mean": round(statistics.mean(values), 4), "min": round(min(values), 4), "max": round(max(values), 4)}


def speed(conn, docs, vectors):
    """Milliseconds per question. Every search runs once per round, so a slower moment affects them all alike;
    round 0 isn't timed, so caches are warm, and garbage collection is off while timing, as in Python's timeit."""
    searches = [("in memory (numpy, steps 3 and 4)", lambda: None, lambda v: search(v[None, :], docs, DEPTH)),
                ("database: sending the question only (no search)", lambda: db.use_planner(conn),
                 lambda v: conn.run("SELECT CAST(:v AS vector) IS NOT NULL", v=db.vector_text(v))),
                ("database: exact", lambda: db.use_planner(conn), lambda v: db.nearest(conn, v, DEPTH, exact=True))]
    searches += [(f"database: HNSW, {ef_name(ef)}", lambda ef=ef: db.use_index(conn, ef),
                  lambda v: db.nearest(conn, v, DEPTH, exact=False)) for ef in EF_SEARCH]
    timings = {name: [] for name, _, _ in searches}
    gc.disable()
    try:
        for round_ in range(TIMED_ROUNDS + 1):
            for name, prepare, run in searches:
                prepare()
                for v in vectors:
                    start = time.perf_counter()
                    run(v)
                    if round_:
                        timings[name].append((time.perf_counter() - start) * 1000)
    finally:
        gc.enable()
    return {name: {"median": round(statistics.median(ms), 3), "p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 3),
                   "timings": len(ms)} for name, ms in timings.items()}


def main():
    started = time.time()
    controls = active_controls()
    keys = {name: load() for name, (_, load) in KEYS.items()}
    docs, qvec = load_embeddings(controls, keys)
    committed = json.loads((ROOT / "results" / "vector_top100.json").read_text())[VECTOR_TEXT]
    published = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())["test"]
    rerank = json.loads((ROOT / "results" / "rerank.json").read_text())
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]
    weight = rerank["chosen"]["vector_weight"]
    methods = ["vector", "hybrid (step 3)", f"hybrid, vector weight {weight:g} (step 4)"]

    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    keyword = {name: {q.id: [controls[i].id for i, _ in bm25.search(q.text, DEPTH)] for q in qs} for name, qs in keys.items()}

    def rankings(vector):
        """The vector search and the hybrid searches of steps 3 and 4, with these vector lists in place of the committed ones."""
        return {methods[0]: vector,
                methods[1]: {n: {q: rrf([keyword[n][q], vector[n][q]], k=60, depth=DEPTH) for q in keyword[n]} for n in keys},
                methods[2]: {n: {q: rrf([keyword[n][q], vector[n][q]], k=60, depth=DEPTH, weights=[1.0, weight])
                                 for q in keyword[n]} for n in keys}}

    def scores(ranked):
        return {n: {m: score(ranked[m][n], qs) for m in methods} for n, qs in keys.items()}

    conn = db.connect_for_checks()
    t = time.perf_counter()
    db.create_table(conn)
    db.load(conn, controls, docs)
    load_seconds = time.perf_counter() - t
    probe = qvec["csf2"][keys["csf2"][0].id]

    # 2. Exact search, against the committed in-memory lists
    db.use_planner(conn)
    exact = {n: {q: db.nearest(conn, v, DEPTH, exact=True) for q, v in qvec[n].items()} for n in keys}
    swaps, same10, same100 = [], 0, 0
    for n in keys:
        for q, v in qvec[n].items():
            got, want = exact[n][q], committed[n][q]
            same10 += got[:10] == want[:10]
            same100 += got == want
            ranks = [r + 1 for r in range(DEPTH) if got[r] != want[r]]
            if ranks:
                similarity = dict(zip((c.id for c in controls), docs.astype(np.float64) @ v.astype(np.float64)))
                swaps.append({"key": n, "question": q, "ranks": ranks,
                              "similarity_gap": float(f"{rank_gap(got, want, similarity):.3g}")})
    exact_ranked, committed_ranked = rankings(exact), rankings(committed)
    exact_scores = scores(exact_ranked)
    reproduced = {
        methods[0]: all(exact_scores[n][methods[0]] == published[n]["vector"] for n in keys),
        methods[1]: all(exact_scores[n][methods[1]] == published[n]["hybrid"] for n in keys),
        methods[2]: all(exact_scores[n][methods[2]] == rerank["test"][n][f"hybrid, vector weight {weight:g}"] for n in keys),
        "reranker candidates, the fused top 30 (step 4)": all(
            {q: r[:30] for q, r in exact_ranked[methods[2]][n].items()} == candidates[n] for n in keys),
    }
    same_fused = {m: sum(exact_ranked[m][n][q][:10] == committed_ranked[m][n][q][:10] for n in keys for q in keyword[n])
                  for m in methods[1:]}

    # 3. The HNSW index: the planner's own choice once it exists, then the index forced, over several builds
    index_seconds, planner = [], {}
    runs = {ef_name(ef): [] for ef in EF_SEARCH}
    for build in range(BUILDS):
        t = time.perf_counter()
        db.create_index(conn)
        index_seconds.append(time.perf_counter() - t)
        if build == 0:
            db.use_planner(conn)
            planner = {f"top {k}": db.scans(conn, probe, k, exact=False)[0][0] for k in (10, DEPTH)}
            db.use_index(conn)  # the settings most in favor of the index
            exact_scans = {f"top {k}": db.scans(conn, probe, k, exact=True) for k in (10, DEPTH)}
            assert all(name != db.INDEX for s in exact_scans.values() for _, name in s), "exact search used the index"
        for ef in EF_SEARCH:
            db.use_index(conn, ef)
            assert db.scans(conn, probe, DEPTH, exact=False) == [("Index Scan", db.INDEX)], "the index wasn't used"
            found = {n: {q: db.nearest(conn, v, DEPTH, exact=False) for q, v in qvec[n].items()} for n in keys}
            ranked = rankings(found)
            runs[ef_name(ef)].append({
                "rows": sorted({len(r) for n in keys for r in found[n].values()}),
                "recall@10": statistics.mean(overlap(found[n][q], exact[n][q], 10) for n in keys for q in found[n]),
                "recall@100": statistics.mean(overlap(found[n][q], exact[n][q], 100) for n in keys for q in found[n]),
                "changed": {m: sum(ranked[m][n][q][:10] != exact_ranked[m][n][q][:10] for n in keys for q in keyword[n])
                            for m in methods[1:]},
                "scores": scores(ranked)})
    sizes = conn.run(f"SELECT pg_total_relation_size('controls'), pg_relation_size('{db.INDEX}')")[0]

    hnsw = {}
    for name, builds in runs.items():
        changed_scores = {}
        for n in keys:
            for m in methods:
                for metric in SHOWN:
                    values = [b["scores"][n][m][metric] for b in builds]
                    if any(v != exact_scores[n][m][metric] for v in values):
                        changed_scores.setdefault(n, {}).setdefault(m, {})[metric] = {
                            "exact": exact_scores[n][m][metric], "min": min(values), "max": max(values)}
        hnsw[name] = {"rows_returned_for_top_100": sorted({r for b in builds for r in b["rows"]}),
                      "recall@10": spread([b["recall@10"] for b in builds]),
                      "recall@100": spread([b["recall@100"] for b in builds]),
                      "hybrid_top_10_changed": {m: {"min": min(b["changed"][m] for b in builds), "max": max(b["changed"][m] for b in builds)}
                                                for m in methods[1:]},
                      "test_scores_that_changed": changed_scores}

    # 4. Speed, on the last index build. Each question asks for its top 100, as the hybrid search does.
    latency = speed(conn, docs, [v for n in keys for v in qvec[n].values()])
    info = db.versions(conn)
    conn.close()

    results = {
        "database": {**info, "image": IMAGE, "driver": "pg8000 1.31.5", "distance": "inner product (the embeddings are unit length)",
                     "hnsw": {**db.HNSW, "operator_class": "vector_ip_ops"}},
        "machine": f"{ {'Darwin': 'macOS'}.get(platform.system(), platform.system())} {platform.machine()}",
        "load": {"controls": len(controls), "load_seconds": round(load_seconds, 2), "index_build_seconds": spread(index_seconds),
                 "table_with_indexes_bytes": sizes[0], "hnsw_index_bytes": sizes[1]},
        "exact": {"questions": sum(len(qs) for qs in keys.values()), "same_top_10": same10, "same_top_100": same100,
                  "swaps": swaps, "same_fused_top_10": same_fused, "published_test_scores_reproduced": reproduced,
                  "test_scores": exact_scores, "scans_with_the_index_forced": exact_scans},
        "planner_with_the_index_not_forced": planner,
        "hnsw": {"builds": BUILDS, "settings": hnsw},
        "methods": methods,
        "speed": {"timed_rounds": TIMED_ROUNDS, "milliseconds_per_question_top_100": latency},
    }
    (ROOT / "results" / "database.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    doc = render_doc(results)
    (ROOT / "docs" / "results-database.md").write_text(doc, encoding="utf-8")
    print(doc)
    print(f"{time.time() - started:.0f} s")


def changes(entry):
    """One line per search whose test scores changed: each changed score across the builds, and the exact one."""
    lines = []
    for n, by_method in entry["test_scores_that_changed"].items():
        for m, by_metric in by_method.items():
            parts = []
            for metric, v in by_metric.items():
                f = (lambda x: f"{x:.3f}") if metric == "mrr" else (lambda x: f"{x:.1%}")
                became = f"{f(v['min'])} in every build" if v["min"] == v["max"] else f"{f(v['min'])} to {f(v['max'])} across builds"
                parts.append(f"{SHOWN[metric]}, {became} (exact: {f(v['exact'])})")
            lines.append(f"{KEYS[n][0]}, {m}: " + "; ".join(parts) + ".")
    return lines


def render_doc(r):
    """docs/results-database.md, from results/database.json alone (a test checks the committed page matches)."""
    e, h, ld, methods = r["exact"], r["hnsw"]["settings"], r["load"], r["methods"]

    def low_high(d):
        return str(d["min"]) if d["min"] == d["max"] else f"{d['min']} to {d['max']}"

    index_rows = "\n".join(
        f"| {name} | {', '.join(map(str, s['rows_returned_for_top_100']))} | {s['recall@10']['mean']:.2%} ({s['recall@10']['min']:.2%}) | "
        f"{s['recall@100']['mean']:.2%} ({s['recall@100']['min']:.2%}) | "
        f"{low_high(s['hybrid_top_10_changed'][methods[1]])} | {low_high(s['hybrid_top_10_changed'][methods[2]])} |"
        for name, s in h.items())
    change_rows = "\n".join(f"- **{name}:**" + ("".join(f"\n  - {line}" for line in changes(s)) or " none.") for name, s in h.items())
    speed_rows = "\n".join(f"| {name} | {m['median']:.2f} | {m['p95']:.2f} |" for name, m in r["speed"]["milliseconds_per_question_top_100"].items())
    swap_note = (f" In the other {len(e['swaps'])}, two neighbors swapped places, between ranks {min(min(s['ranks']) for s in e['swaps'])} "
                 f"and {max(max(s['ranks']) for s in e['swaps'])}. Their similarities differ by at most "
                 f"{max(s['similarity_gap'] for s in e['swaps']):.2g}, which is float32 rounding." if e["swaps"] else "")
    missing = [name for name, ok in e["published_test_scores_reproduced"].items() if not ok]
    reproduced = ("**Every published test score comes back:** vector search, both hybrid searches, and the reranker's 30 "
                  "candidates per question, so the reranked scores too." if not missing else
                  "**Not every published test score comes back.** Different: " + "; ".join(missing) + ".")
    doc = f"""# Results: the vector database (step 5)

Generated by `scripts/run_database_check.py`. Earlier steps: [keyword search](results.md), [vector and hybrid search](results-vector-hybrid.md), [weighting and reranking](results-rerank.md).

## Setup
- **Database:** PostgreSQL {r['database']['postgresql'].split()[0]} with pgvector {r['database']['pgvector']}, in Docker (`docker-compose.yml`, the image pinned by digest). The driver is pg8000, written in pure Python.
- **Table:** one row per active control ({ld['controls']:,}), with its text and its embedding in a `vector(768)` column. The embeddings are the ones behind steps 3 and 4, committed in `embeddings/`. They're unit length, so the search uses the inner product, which is then the cosine similarity.
- **Searches:** exact, which orders by pgvector's `inner_product()` function, so every control is compared (no index can serve that order), with ties broken by catalog order like the in-memory search; and pgvector's HNSW index, through the `<#>` operator (m = {r['database']['hnsw']['m']}, ef_construction = {r['database']['hnsw']['ef_construction']}, its defaults). The index places each control at a random level, so it was built {r['hnsw']['builds']} times.
- **Loading:** {ld['load_seconds']:.1f} s for the {ld['controls']:,} rows, one insert at a time in one transaction, and {ld['index_build_seconds']['mean']:.2f} s to build the index. The table and its indexes take {ld['table_with_indexes_bytes'] / 1e6:.1f} MB, {ld['hnsw_index_bytes'] / 1e6:.1f} MB of it the HNSW index.

## Exact search gives the same results as the in-memory search
- **The same top 10** for {e['same_top_10']} of {e['questions']} questions, and the same top 100 for {e['same_top_100']}.{swap_note}
- {reproduced}

## The HNSW index
- **PostgreSQL decides when to use the index.** With the index in place and nothing forced, its planner used the index for a top 10 ({r['planner_with_the_index_not_forced']['top 10']}), so that result was approximate, and scanned the whole table for a top 100 ({r['planner_with_the_index_not_forced']['top 100']}), which is exact. The numbers below force the index, and each setting's plan was checked; the exact search never used it, even forced.
- **pgvector's default returns only 40 rows.** It keeps ef_search candidates (40 by default) and returns no more than that, so a top-100 request silently gets 40.

| Setting | Rows returned for a top 100 | Exact top 10 found: mean (lowest build) | Exact top 100 found | Questions whose top 10 changed: {methods[1]} | Changed: {methods[2]} |
|---|---|---|---|---|---|
{index_rows}

Test scores that changed (across the {r['hnsw']['builds']} builds):
{change_rows}

## Speed
Milliseconds per question, asking for its top 100, measured from Python on this computer ({r['machine']}, the database in Docker). Each search ran {r['speed']['timed_rounds']} timed rounds of all {e['questions']} questions.

| Search | Median (ms) | 95th percentile (ms) |
|---|---|---|
{speed_rows}

## Reading these numbers
- **Exact top k found:** the share of the exact search's top k that the index also returned.
- **Hybrid top 10 changed:** questions whose fused top 10 differs from the one built from exact search.
- **Timings** depend on the computer and include the trip to the database and back, so CI doesn't check them. CI does check the exact search, the rows returned, and the planner's choices against a database of its own.
"""
    return doc


if __name__ == "__main__":
    main()
