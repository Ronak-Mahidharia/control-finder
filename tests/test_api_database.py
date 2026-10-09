# The API against a real database, with the committed question embeddings standing in for the model
# (CI has no Ollama): every question's vector and hybrid answers must be the published ones.
# Runs when CONTROLFINDER_DB_URL is set, like tests/test_database.py.
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.api import create_app
from controlfinder.metrics import summarize
from controlfinder.search import Pool, Searcher

pytestmark = pytest.mark.skipif("CONTROLFINDER_DB_URL" not in os.environ, reason="no database (set CONTROLFINDER_DB_URL)")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "embeddings" / "manifest.json").read_text())
VECTOR = json.loads((ROOT / "results" / "vector_top100.json").read_text())[MANIFEST["controls"]["text"]]
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}


@pytest.fixture(scope="module")
def client():
    embedded = {}
    for key, load in KEYS.items():
        texts = {q.id: q.text for q in load()}
        entry = MANIFEST["questions"][key]
        embedded.update(zip((texts[i] for i in entry["ids"]), np.load(ROOT / "embeddings" / entry["file"])))
    app = create_app(Searcher(Pool(), embed=embedded.__getitem__), embedding_model_check=lambda: True)
    with TestClient(app) as c:
        yield c


def test_health_with_the_database(client):
    body = client.get("/health").json()
    published = json.loads((ROOT / "results" / "database.json").read_text())["database"]
    assert body["status"] == "ok" and body["database"] in ("ready", "loaded")
    assert body["pgvector"] == published["pgvector"]


def test_every_answer_is_the_published_one(client):
    vector_hybrid = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())["test"]
    rerank = json.loads((ROOT / "results" / "rerank.json").read_text())
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]
    weight = rerank["chosen"]["vector_weight"]
    for key, load in KEYS.items():
        questions = load()
        vector, hybrid = {}, {}
        for q in questions:
            vector[q.id] = [r["id"] for r in client.get("/search", params={"q": q.text, "method": "vector"}).json()["results"]]
            results = client.get("/search", params={"q": q.text, "k": 30}).json()["results"]
            hybrid[q.id] = [r["id"] for r in results]
            assert vector[q.id] == VECTOR[key][q.id][:10], q.id
            assert hybrid[q.id] == candidates[key][q.id], q.id
            for r in results:
                assert r["vector_rank"] == (VECTOR[key][q.id].index(r["id"]) + 1 if r["id"] in VECTOR[key][q.id] else None), q.id
        test = [q for q in questions if q.split == "test"]
        assert summarize([(vector[q.id], q.expected) for q in test]) == vector_hybrid[key]["vector"]
        assert summarize([(hybrid[q.id][:10], q.expected) for q in test]) == rerank["test"][key][f"hybrid, vector weight {weight:g}"]


def test_the_timing_header_names_each_part(client):
    q = load_csf_questions()[0].text
    timing = client.get("/search", params={"q": q}).headers["server-timing"]
    assert [part.split(";")[0] for part in timing.split(", ")] == ["keyword", "embed", "vector", "fuse"]
