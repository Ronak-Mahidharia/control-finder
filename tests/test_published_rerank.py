# Step 4's published numbers must come back from committed data alone: the keyword search
# (recomputed), the step 3 vector lists, and every candidate's reranker score. CI has no PyTorch.
import importlib.util
import json
from pathlib import Path

import pytest

from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.fusion import rrf
from controlfinder.metrics import summarize
from controlfinder.rerank import reorder

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "rerank.json").read_text())
SCORES = json.loads((ROOT / "results" / "rerank_scores.json").read_text())
VECTOR = json.loads((ROOT / "results" / "vector_top100.json").read_text())[RESULTS["settings"]["vector_text"]]
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}


def score(rankings, questions, split):
    return summarize([(rankings[q.id][:10], q.expected) for q in questions if q.split == split])


@pytest.fixture(scope="module")
def keyword():
    controls = active_controls()
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    return {key: {q.id: [controls[i].id for i, _ in bm25.search(q.text, 100)] for q in load()} for key, load in KEYS.items()}


def fused(keyword, key, weight):
    return {qid: rrf([keyword[key][qid], VECTOR[key][qid]], k=60, depth=100, weights=[1.0, weight]) for qid in keyword[key]}


def test_weight_was_chosen_on_csf_dev(keyword):
    csf = load_csf_questions()
    for w in RESULTS["settings"]["weights_tried"]:
        assert score(fused(keyword, "csf2", w), csf, "dev") == RESULTS["dev_csf2"]["weights"][f"hybrid, vector weight {w:g}"]
    dev = RESULTS["dev_csf2"]["weights"]
    best = max(RESULTS["settings"]["weights_tried"],
               key=lambda w: (dev[f"hybrid, vector weight {w:g}"]["mrr"], dev[f"hybrid, vector weight {w:g}"]["recall@10"], -w))
    assert RESULTS["chosen"]["vector_weight"] == best


def test_candidates_are_the_fused_top_30(keyword):
    for key in KEYS:
        first = fused(keyword, key, RESULTS["chosen"]["vector_weight"])
        assert {qid: ranking[:30] for qid, ranking in first.items()} == SCORES["candidates"][key]


def test_reranker_was_chosen_on_csf_dev():
    csf = [q for q in load_csf_questions() if q.split == "dev"]
    dev = {}
    for model, by_question in SCORES["dev_scores_csf2"].items():
        reranked = {q.id: reorder(SCORES["candidates"]["csf2"][q.id], by_question[q.id]) for q in csf}
        dev[model] = score(reranked, csf, "dev")
        assert dev[model] == RESULTS["dev_csf2"]["rerankers"][f"reranked: {model}"]
    assert RESULTS["chosen"]["reranker"] == max(dev, key=lambda m: (dev[m]["mrr"], dev[m]["recall@10"]))


def test_test_scores_reproduce(keyword):
    weight, model = RESULTS["chosen"]["vector_weight"], RESULTS["chosen"]["reranker"]
    for key, load in KEYS.items():
        questions = load()
        reranked = {qid: reorder(c, SCORES["chosen_scores"][key][qid]) for qid, c in SCORES["candidates"][key].items()}
        published = RESULTS["test"][key]
        assert score(fused(keyword, key, 1.0), questions, "test") == published["hybrid, vector weight 1 (step 3)"]
        assert score(fused(keyword, key, weight), questions, "test") == published[f"hybrid, vector weight {weight:g}"]
        assert score(reranked, questions, "test") == published[f"reranked: {model}"]


def test_readme_numbers_match():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    weight, model = RESULTS["chosen"]["vector_weight"], RESULTS["chosen"]["reranker"]
    for key, label in (("csf2", "CSF 2.0"), ("sp800_171", "SP 800-171")):
        for method, name in ((f"hybrid, vector weight {weight:g}", f"Hybrid, vector weight {weight:g}"),
                             (f"reranked: {model}", "Hybrid + reranker")):
            m = RESULTS["test"][key][method]
            row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |") and f"| {name}" in line)
            assert f"| {m['mrr']:.3f} |" in row and f"| {m['hit@1'] * 100:.1f}% |" in row and f"| {m['hit@10'] * 100:.1f}% |" in row, row


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None,
                    reason="the reranker isn't installed (CI checks the committed scores instead)")
def test_reranker_gives_the_committed_scores_again():
    # Scored in the same batches as the published run, the scores are identical. Scored one question
    # at a time, the batches differ, which changes the padding and the last digits of the arithmetic
    # (by up to about 0.002 on a range of about -3.6 to 0.3, checked Oct 8, 2026). So this allows 0.01
    # and also requires the same order.
    from controlfinder.rerank import load, score_pairs

    by_id = {c.id: c for c in active_controls()}
    model_name = RESULTS["chosen"]["reranker"]
    model = load(model_name)
    questions = {q.id: q for q in load_csf_questions()}
    for qid in sorted(SCORES["dev_scores_csf2"][model_name])[:3]:
        candidates = SCORES["candidates"]["csf2"][qid]
        saved = SCORES["dev_scores_csf2"][model_name][qid]
        again = score_pairs(model, [(questions[qid].text, f"{by_id[c].title}. {by_id[c].statement}") for c in candidates])
        assert max(abs(a - b) for a, b in zip(again, saved)) < 0.01, qid
        assert reorder(candidates, again) == reorder(candidates, saved), qid
