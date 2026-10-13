# The request records and GET /stats (src/controlfinder/trace.py), through the real API with no database and a
# stand-in embedding model and answer model.
import json

import numpy as np
from fastapi.testclient import TestClient

from controlfinder.api import create_app
from controlfinder.search import InMemorySearcher, Searcher
from controlfinder.trace import Tracer, percentile, request_id

from test_api import NoDatabase, no_model

QUESTION = "How should we store passwords?"


def unit(question):
    v = np.ones(768, dtype=np.float32)
    return v / np.linalg.norm(v)


class MemorySearcher(InMemorySearcher):
    """The vectors searched in memory, and no database for the API to reach at startup."""

    def __init__(self):
        super().__init__(embed=unit)
        self.pool = NoDatabase()


def stand_in_model(question, shown):
    return {"raw": json.dumps({"covered": True, "answer": "Hash them.", "citations": [shown[0].label]}),
            "seconds": 0.01, "prompt_tokens": 812, "output_tokens": 41}


def test_the_nearest_rank_percentile():
    assert percentile([5, 1, 3, 2, 4], 0.5) == 3 and percentile([5, 1, 3, 2, 4], 0.95) == 5
    assert percentile([7], 0.95) == 7 and percentile([], 0.5) is None


def test_a_callers_request_id_is_kept_only_if_its_safe_to_send_back():
    assert request_id("abc-123_X") == "abc-123_X"
    for unsafe in (None, "", "has space", "x" * 65, "new\nline", "<script>"):
        made = request_id(unsafe)
        assert len(made) == 16 and all(c in "0123456789abcdef" for c in made)


def test_each_request_gets_a_record_with_its_steps_and_never_the_question(tmp_path):
    tracer = Tracer(tmp_path / "trace.jsonl")
    with TestClient(create_app(MemorySearcher(), embedding_model_check=lambda: True, site=None,
                               tracer=tracer, answerer=stand_in_model)) as c:
        sent = c.get("/search", params={"q": QUESTION, "k": 5}, headers={"X-Request-ID": "mine-1"})
        assert sent.status_code == 200 and sent.headers["X-Request-ID"] == "mine-1"
        assert c.get("/answer", params={"q": QUESTION}).status_code == 200
        assert c.get("/controls/IA-5(1)").status_code == 200 and c.get("/controls/nope").status_code == 404
        stats = c.get("/stats").json()
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["route"] for r in records] == ["/search", "/answer", "/controls/{control_id}", "/controls/{control_id}"]
    assert [r["status"] for r in records] == [200, 200, 200, 404]  # /stats itself isn't recorded
    search = records[0]
    assert search["id"] == "mine-1" and search["method"] == "hybrid" and search["question_chars"] == len(QUESTION)
    assert set(search["steps"]) == {"keyword", "embed", "vector", "fuse"} and search["results"] == 5
    assert records[1]["prompt_tokens"] == 812 and records[1]["output_tokens"] == 41 and "answer" in records[1]["steps"]
    assert all(QUESTION not in line and "passwords" not in line for line in lines)
    assert stats["since_start"] == {"requests": 4, "errors": 0} and stats["window"] == 4
    assert stats["routes"]["/search"]["requests"] == 1 and stats["routes"]["/controls/{control_id}"]["requests"] == 2
    assert stats["steps_ms"]["embed"]["count"] == 2 and stats["answer_tokens"]["prompt_tokens"]["p50"] == 812


def test_a_service_thats_down_is_counted_as_an_error_with_its_reason():
    tracer = Tracer()
    with TestClient(create_app(Searcher(NoDatabase(), embed=no_model), embedding_model_check=lambda: False, site=None,
                               tracer=tracer)) as c:
        assert c.get("/search", params={"q": QUESTION, "method": "vector"}).status_code == 503
        assert c.get("/search", params={"q": QUESTION, "method": "keyword"}).status_code == 200
        stats = c.get("/stats").json()
    assert stats["since_start"] == {"requests": 2, "errors": 1}
    failed = tracer.recent[0]
    assert failed["status"] == 503 and "embedding model" in failed["error"]


def test_only_the_last_window_of_records_is_kept_in_memory():
    tracer = Tracer(window=3)
    for n in range(5):
        tracer.record({"route": "/health", "status": 200, "ms": float(n)})
    stats = tracer.stats()
    assert stats["since_start"]["requests"] == 5 and stats["window"] == 3
    assert stats["routes"]["/health"]["ms"] == {"p50": 3.0, "p95": 4.0, "count": 3}
