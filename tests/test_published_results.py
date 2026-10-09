# The published baseline numbers must come back exactly when the search is run again.
import json
from pathlib import Path

from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.metrics import summarize

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS = json.loads((RESULTS_DIR / "keyword_baseline.json").read_text())
RESULTS_CSF = json.loads((RESULTS_DIR / "keyword_baseline_csf2.json").read_text())


def test_published_test_scores_reproduce():
    controls = active_controls()
    chosen = RESULTS["chosen"]
    assert chosen == "title + statement + guidance"
    index = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    test = [q for q in load_questions() if q.split == "test"]
    ranked = [([controls[i].id for i, _ in index.search(q.text, 10)], q.expected) for q in test]
    assert summarize(ranked) == RESULTS["test"]
    assert all(RESULTS["rankings"][q.id]["top10"] == r for q, (r, _) in zip(test, ranked))


def test_published_csf_scores_reproduce():
    controls = active_controls()
    assert RESULTS_CSF["chosen"] == "title + statement + guidance"
    index = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    test = [q for q in load_csf_questions() if q.split == "test"]
    ranked = [([controls[i].id for i, _ in index.search(q.text, 10)], q.expected) for q in test]
    assert summarize(ranked) == RESULTS_CSF["test"]
    assert RESULTS_CSF["test"]["mrr"] < RESULTS["test"]["mrr"] / 2  # the CSF key really is harder for keywords
