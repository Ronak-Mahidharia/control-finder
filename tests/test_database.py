# Step 5 against a real database. These run when CONTROLFINDER_DB_URL is set: CI starts one, and
# locally `docker compose up -d --wait` starts the one at db.LOCAL_URL. They rebuild the controls table.
import json
import os
import statistics
from pathlib import Path

import numpy as np
import pytest

from controlfinder import db
from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.fusion import rrf
from controlfinder.metrics import summarize
from controlfinder.vector import rank_gap

pytestmark = pytest.mark.skipif("CONTROLFINDER_DB_URL" not in os.environ, reason="no database (set CONTROLFINDER_DB_URL)")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "embeddings" / "manifest.json").read_text())
COMMITTED = json.loads((ROOT / "results" / "vector_top100.json").read_text())[MANIFEST["controls"]["text"]]
PUBLISHED = json.loads((ROOT / "results" / "database.json").read_text())
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}
DOCS = np.load(ROOT / "embeddings" / MANIFEST["controls"]["file"])
QUESTIONS = {key: dict(zip(MANIFEST["questions"][key]["ids"], np.load(ROOT / "embeddings" / MANIFEST["questions"][key]["file"])))
             for key in KEYS}
PROBE = QUESTIONS["csf2"][MANIFEST["questions"]["csf2"]["ids"][0]]


@pytest.fixture(scope="module")
def conn():
    connection = db.connect()
    db.create_table(connection)
    db.load(connection, active_controls(), DOCS)
    db.create_index(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def exact(conn):
    db.use_planner(conn)
    return {key: {qid: db.nearest(conn, v, 100, exact=True) for qid, v in QUESTIONS[key].items()} for key in KEYS}


def test_exact_search_gives_the_in_memory_lists(exact):
    ids = MANIFEST["controls"]["ids"]
    for key in KEYS:
        for qid, v in QUESTIONS[key].items():
            got, want = exact[key][qid], COMMITTED[key][qid]
            assert got[:10] == want[:10], qid
            assert rank_gap(got, want, dict(zip(ids, DOCS.astype(np.float64) @ v.astype(np.float64)))) < 1e-6, qid


def test_published_scores_come_back_from_the_database(exact):
    controls = active_controls()
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    vector_hybrid = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())["test"]
    rerank = json.loads((ROOT / "results" / "rerank.json").read_text())
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]
    weight = rerank["chosen"]["vector_weight"]
    for key, load in KEYS.items():
        questions = load()
        test = [q for q in questions if q.split == "test"]
        keyword = {q.id: [controls[i].id for i, _ in bm25.search(q.text, 100)] for q in questions}
        plain = {q.id: rrf([keyword[q.id], exact[key][q.id]], k=60, depth=100) for q in questions}
        weighted = {q.id: rrf([keyword[q.id], exact[key][q.id]], k=60, depth=100, weights=[1.0, weight]) for q in questions}
        assert summarize([(exact[key][q.id][:10], q.expected) for q in test]) == vector_hybrid[key]["vector"]
        assert summarize([(plain[q.id][:10], q.expected) for q in test]) == vector_hybrid[key]["hybrid"]
        assert summarize([(weighted[q.id][:10], q.expected) for q in test]) == rerank["test"][key][f"hybrid, vector weight {weight:g}"]
        assert {qid: ranking[:30] for qid, ranking in weighted.items()} == candidates[key]


def test_planner_choices_are_the_published_ones(conn):
    db.use_planner(conn)
    for k in (10, 100):
        assert db.scans(conn, PROBE, k, exact=False)[0][0] == PUBLISHED["planner_with_the_index_not_forced"][f"top {k}"]
    db.use_index(conn)  # the settings most in favor of the index: exact search must still compare every control
    for k in (10, 100):
        assert all(name != db.INDEX for _, name in db.scans(conn, PROBE, k, exact=True))
    assert db.scans(conn, PROBE, 100, exact=False) == [("Index Scan", db.INDEX)]


def test_default_ef_search_returns_only_40_rows(conn):
    for ef, published in (("ef_search 40 (default)", None), ("ef_search 100", 100), ("ef_search 200", 200)):
        db.use_index(conn, published)
        rows = {len(db.nearest(conn, v, 100, exact=False)) for v in list(QUESTIONS["csf2"].values())[:20]}
        assert sorted(rows) == PUBLISHED["hnsw"]["settings"][ef]["rows_returned_for_top_100"], ef
    assert PUBLISHED["hnsw"]["settings"]["ef_search 40 (default)"]["rows_returned_for_top_100"] == [40]


def test_index_finds_the_exact_top_10_at_ef_search_100(conn, exact):
    # Every published build found 100% of it. Each build places the controls at random levels,
    # so this allows a little less.
    db.use_index(conn, 100)
    found = [len(set(db.nearest(conn, v, 10, exact=False)) & set(exact[key][qid][:10])) / 10
             for key in KEYS for qid, v in QUESTIONS[key].items()]
    assert statistics.mean(found) >= 0.99
