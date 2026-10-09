# The API without a database or the embedding model: input checks, control lookups, keyword search,
# and what happens when vector search can't run. tests/test_api_database.py checks the rest.
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pg8000.exceptions import InterfaceError

from controlfinder import search as search_module
from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.api import MAX_K, MAX_QUESTION, create_app
from controlfinder.catalog import active_controls
from controlfinder.embeddings import QUERY
from controlfinder.metrics import summarize
from controlfinder.search import Pool, Searcher, Unavailable

ROOT = Path(__file__).resolve().parent.parent
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}


class NoDatabase:
    @contextmanager
    def connection(self):
        raise InterfaceError("no database in this test")
        yield

    def close(self):
        pass


def no_model(question):
    raise Unavailable("no embedding model in this test")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Searcher(NoDatabase(), embed=no_model), embedding_model_check=lambda: False)) as c:
        yield c


def test_health_says_what_is_missing(client):
    body = client.get("/health").json()
    assert body == {"status": "degraded", "controls": 1014, "database": "unreachable", "postgresql": None,
                    "pgvector": None, "embedding_model": "unreachable"}


def test_keyword_search_gives_the_published_scores(client):
    published = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())["test"]
    for key, load in KEYS.items():
        test = [q for q in load() if q.split == "test"]
        ranked = [([r["id"] for r in client.get("/search", params={"q": q.text, "method": "keyword"}).json()["results"]], q.expected)
                  for q in test]
        assert summarize(ranked) == published[key]["keyword (BM25)"], key


def test_results_carry_each_controls_label_title_and_ranks(client):
    body = client.get("/search", params={"q": "store passwords", "method": "keyword", "k": 3}).json()
    by_id = {c.id: c for c in active_controls()}
    assert body["method"] == "keyword" and body["vector_weight"] is None
    for i, r in enumerate(body["results"], start=1):
        c = by_id[r["id"]]
        assert (r["rank"], r["keyword_rank"], r["vector_rank"]) == (i, i, None)
        assert (r["label"], r["title"], r["family"]) == (c.label, c.title, c.family)


def test_vector_and_hybrid_answer_503_and_point_to_keyword_search(client):
    for method in ("vector", "hybrid"):
        response = client.get("/search", params={"q": "store passwords", "method": method})
        assert response.status_code == 503 and "method=keyword" in response.json()["detail"], method


def test_answers_need_the_search_and_say_so(client):
    response = client.get("/answer", params={"q": "store passwords"})
    assert response.status_code == 503 and "method=keyword" in response.json()["detail"]
    for bad in ({}, {"q": "  "}, {"q": "a" * (MAX_QUESTION + 1)}):
        assert client.get("/answer", params=bad).status_code == 422, bad


def test_bad_input_is_refused(client):
    ok = {"q": "store passwords", "method": "keyword"}
    assert client.get("/search", params=ok).status_code == 200
    assert client.get("/search", params={**ok, "q": "a" * MAX_QUESTION}).status_code == 200
    for bad in ({"method": "keyword"}, {**ok, "q": ""}, {**ok, "q": "   "}, {**ok, "q": "a" * (MAX_QUESTION + 1)},
                {**ok, "k": 0}, {**ok, "k": MAX_K + 1}, {**ok, "method": "bm25"}):
        assert client.get("/search", params=bad).status_code == 422, bad


def test_a_control_can_be_read_by_its_id(client):
    body = client.get("/controls/IA-5.1").json()  # ids are matched without regard to case
    assert (body["id"], body["label"], body["title"], body["parent"]) == ("ia-5.1", "IA-5(1)", "Password-based Authentication", "ia-5")
    assert body["statement"] and body["assessment_objectives"]
    withdrawn = client.get("/controls/ac-2.10")
    assert withdrawn.status_code == 404 and withdrawn.json()["detail"] == "AC-2(10) is withdrawn in SP 800-53 Rev 5.2.0"
    assert client.get("/controls/zz-1").status_code == 404


def test_settings_are_the_published_ones():
    rerank = json.loads((ROOT / "results" / "rerank.json").read_text())
    settings = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())["settings"]
    assert search_module.VECTOR_WEIGHT == rerank["chosen"]["vector_weight"]
    assert (search_module.RRF_K, search_module.DEPTH) == (settings["rrf_k"], settings["fusion_depth"])


@pytest.fixture
def fresh_cache():
    search_module._embedded.cache_clear()
    yield
    search_module._embedded.cache_clear()


def test_a_question_is_embedded_once_with_the_query_prefix(monkeypatch, fresh_cache):
    calls = []

    def fake_embed(texts, prefix):
        calls.append((tuple(texts), prefix))
        return np.array([[0.6, 0.8]], dtype=np.float32), 0

    monkeypatch.setattr(search_module, "embed", fake_embed)
    first, again = search_module.embed_question("a question"), search_module.embed_question("a question")
    assert calls == [(("a question",), QUERY)]
    assert first.dtype == np.float32 and first.tolist() == again.tolist() == pytest.approx([0.6, 0.8])


def test_an_unreachable_model_becomes_unavailable(monkeypatch, fresh_cache):
    def refused(texts, prefix):
        raise ConnectionRefusedError("Ollama isn't running")

    monkeypatch.setattr(search_module, "embed", refused)
    with pytest.raises(Unavailable):
        search_module.embed_question("another question")


def test_the_pool_replaces_a_connection_that_failed(monkeypatch):
    made = []

    class FakeConnection:
        def __init__(self):
            self.closed = False
            made.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(search_module.db, "connect", lambda url=None: FakeConnection())
    pool = Pool(size=2)
    with pool.connection() as first:
        pass
    with pool.connection() as reused:  # last in, first out: the open one again
        pass
    with pytest.raises(RuntimeError):
        with pool.connection() as failed:
            raise RuntimeError("the query failed")
    with pool.connection() as fresh:
        pass
    assert first is reused is failed and failed.closed and fresh is not failed and len(made) == 2


def test_the_built_website_is_served_after_the_api(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Control Finder</title>", encoding="utf-8")
    searcher = Searcher(NoDatabase(), embed=no_model)
    with TestClient(create_app(searcher, embedding_model_check=lambda: False, site=tmp_path)) as c:
        page = c.get("/")
        assert page.status_code == 200 and "<title>Control Finder</title>" in page.text
        assert c.get("/health").json()["controls"] == 1014  # the API's routes still come first
        assert c.get("/search", params={"q": "store passwords", "method": "keyword"}).status_code == 200
        assert c.get("/nothing-here").status_code == 404
    with TestClient(create_app(Searcher(NoDatabase(), embed=no_model), embedding_model_check=lambda: False, site=None)) as c:
        assert c.get("/").status_code == 404  # no website unless it has been built
