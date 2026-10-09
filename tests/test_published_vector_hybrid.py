# The published vector and hybrid scores must come back from the committed top-100 vector lists,
# without running the embedding model (CI has no Ollama).
import json
import re
from pathlib import Path

from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.fusion import rrf
from controlfinder.metrics import summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "vector_hybrid.json").read_text())
VECTOR = json.loads((ROOT / "results" / "vector_top100.json").read_text())
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}


def scores(rankings, questions, split="test"):
    return summarize([(rankings[q.id][:10], q.expected) for q in questions if q.split == split])


def test_vector_and_hybrid_scores_reproduce():
    controls = active_controls()
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    vector_text, hybrid_text = RESULTS["chosen"]["vector"], RESULTS["chosen"]["hybrid"]
    for key, load in KEYS.items():
        questions = load()
        keyword = {q.id: [controls[i].id for i, _ in bm25.search(q.text, 100)] for q in questions}
        hybrid = {q.id: rrf([keyword[q.id], VECTOR[hybrid_text][key][q.id]], k=60, depth=100) for q in questions}
        assert scores(keyword, questions) == RESULTS["test"][key]["keyword (BM25)"]
        assert scores(VECTOR[vector_text][key], questions) == RESULTS["test"][key]["vector"]
        assert scores(hybrid, questions) == RESULTS["test"][key]["hybrid"]


def test_choices_were_made_on_csf_dev_by_mrr():
    dev = RESULTS["dev_csf2"]
    for method in ("vector", "hybrid"):
        rows = {name.split(": ", 1)[1]: m for name, m in dev.items() if name.startswith(method + ":")}
        assert RESULTS["chosen"][method] == max(rows, key=lambda t: (rows[t]["mrr"], rows[t]["recall@10"]))


def test_every_vector_list_is_complete_and_valid():
    active = {c.id for c in active_controls()}
    for text, by_key in VECTOR.items():
        for key, load in KEYS.items():
            questions = load()
            assert set(by_key[key]) == {q.id for q in questions}
            for ranking in by_key[key].values():
                assert len(ranking) == 100 and len(set(ranking)) == 100 and set(ranking) <= active


def test_readme_numbers_match_the_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, label in (("csf2", "CSF 2.0"), ("sp800_171", "SP 800-171")):
        for method, name in (("keyword (BM25)", "Keyword"), ("vector", "Vector"), ("hybrid", "Hybrid")):
            m = RESULTS["test"][key][method]
            row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |") and f"| {name}" in line)
            assert f"| {m['mrr']:.3f} |" in row and f"| {m['hit@1'] * 100:.1f}% |" in row and f"| {m['hit@10'] * 100:.1f}% |" in row, row
    assert not re.search(r"\b(?:will|should) (?:beat|improve)\b", readme)  # no claims ahead of the numbers
