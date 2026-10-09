# ============================================================================
# Step 4: weighting the hybrid search, then reranking its top candidates.
#
# Run:   python scripts/run_rerank.py   (needs requirements-rerank.txt)
#   1. First search: keyword (BM25, recomputed) and vector (the committed top
#      100 from step 3), fused with reciprocal rank fusion.
#   2. Weighting: the vector list counts 1, 1.5, 2, or 3 times as much as the
#      keyword list. The weight is chosen on CSF 2.0 dev questions by MRR.
#   3. Reranking: the top 30 of that fused list are rescored by a cross-encoder
#      reading the question with each control's title and statement. Three
#      models are tried on CSF 2.0 dev questions only; the best by MRR is then
#      run once on the test questions of both keys.
#   4. Writes results/rerank.json (scores and choices),
#      results/rerank_scores.json (every candidate's reranker score, so CI can
#      check the rankings without the models), and docs/results-rerank.md.
# Every score uses the top 10, the same as steps 1 to 3.
# ============================================================================
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder.answer_key import load_csf_questions, load_questions  # noqa: E402
from controlfinder.bm25 import BM25  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402
from controlfinder.fusion import rrf  # noqa: E402
from controlfinder.metrics import summarize  # noqa: E402
from controlfinder.rerank import MAX_TOKENS, MODELS, load, reorder, score_pairs  # noqa: E402

VECTOR_TEXT = "title + statement + guidance"  # chosen in step 3
WEIGHTS = [1.0, 1.5, 2.0, 3.0]
CANDIDATES = 30
KEYS = {"csf2": ("CSF 2.0", load_csf_questions), "sp800_171": ("SP 800-171 Rev 3", load_questions)}
METRICS = ["mrr", "hit@1", "hit@5", "hit@10", "recall@10", "precision@5"]


def score(rankings, questions, split):
    chosen = [q for q in questions if q.split == split]
    return summarize([(rankings[q.id][:10], q.expected) for q in chosen])


def table(rows):
    lines = ["| Method | Questions | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for name, m in rows:
        lines.append(f"| {name} | {m['questions']} | " + " | ".join(f"{m[x]:.3f}" for x in METRICS) + " |")
    return "\n".join(lines)


def main():
    started = time.time()
    controls = active_controls()
    by_id = {c.id: c for c in controls}
    keys = {name: load_questions_fn() for name, (_, load_questions_fn) in KEYS.items()}
    vector = json.loads((ROOT / "results" / "vector_top100.json").read_text())[VECTOR_TEXT]
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    keyword = {name: {q.id: [controls[i].id for i, _ in bm25.search(q.text, 100)] for q in qs} for name, qs in keys.items()}

    def fused(name, weight):
        return {qid: rrf([keyword[name][qid], vector[name][qid]], k=60, depth=100, weights=[1.0, weight])
                for qid in keyword[name]}

    csf = keys["csf2"]
    weight_dev = {f"hybrid, vector weight {w:g}": score(fused("csf2", w), csf, "dev") for w in WEIGHTS}
    best_weight = max(WEIGHTS, key=lambda w: (weight_dev[f"hybrid, vector weight {w:g}"]["mrr"],
                                              weight_dev[f"hybrid, vector weight {w:g}"]["recall@10"], -w))
    first = {name: fused(name, best_weight) for name in keys}
    candidates = {name: {qid: ranking[:CANDIDATES] for qid, ranking in first[name].items()} for name in keys}
    recall_at_candidates = {name: round(sum(len(set(candidates[name][q.id]) & q.expected) / len(q.expected)
                                            for q in qs if q.split == "test") / sum(q.split == "test" for q in qs), 3)
                            for name, qs in keys.items()}

    def pairs(name, qids):
        questions = {q.id: q for q in keys[name]}
        return [(questions[qid].text, f"{by_id[c].title}. {by_id[c].statement}") for qid in qids for c in candidates[name][qid]]

    def rerank(model, name, qids):
        flat = score_pairs(model, pairs(name, qids))
        out = {}
        for i, qid in enumerate(qids):
            out[qid] = flat[i * CANDIDATES:(i + 1) * CANDIDATES]
        return out

    dev_ids = [q.id for q in csf if q.split == "dev"]
    model_dev, dev_scores = {}, {}
    for name in MODELS:
        model = load(name)
        dev_scores[name] = rerank(model, "csf2", dev_ids)
        reranked = {qid: reorder(candidates["csf2"][qid], dev_scores[name][qid]) for qid in dev_ids}
        model_dev[name] = score(reranked, [q for q in csf if q.split == "dev"], "dev")
        print(f"dev {name}: {model_dev[name]['mrr']:.3f} ({time.time() - started:.0f} s)")
    best_model = max(MODELS, key=lambda m: (model_dev[m]["mrr"], model_dev[m]["recall@10"]))

    model = load(best_model)
    all_scores = {name: rerank(model, name, [q.id for q in qs]) for name, qs in keys.items()}
    reranked = {name: {qid: reorder(candidates[name][qid], all_scores[name][qid]) for qid in candidates[name]} for name in keys}
    test = {name: {"hybrid, vector weight 1 (step 3)": score(fused(name, 1.0), qs, "test"),
                   f"hybrid, vector weight {best_weight:g}": score(first[name], qs, "test"),
                   f"reranked: {best_model}": score(reranked[name], qs, "test")} for name, qs in keys.items()}

    results = {
        "settings": {"vector_text": VECTOR_TEXT, "weights_tried": WEIGHTS, "candidates": CANDIDATES,
                     "reranker_text": "title + statement", "reranker_max_tokens": MAX_TOKENS,
                     "models": {m: {"revision": r, "license": lic} for m, (r, lic) in MODELS.items()},
                     "chosen_on": "CSF 2.0 dev questions, by MRR"},
        "dev_csf2": {"weights": weight_dev, "rerankers": {f"reranked: {m}": s for m, s in model_dev.items()}},
        "chosen": {"vector_weight": best_weight, "reranker": best_model},
        "test_recall_of_candidates": recall_at_candidates,
        "test": test,
    }
    (ROOT / "results" / "rerank.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (ROOT / "results" / "rerank_scores.json").write_text(json.dumps(
        {"candidates": candidates, "dev_scores_csf2": dev_scores, "chosen_scores": all_scores}, separators=(",", ":")) + "\n", encoding="utf-8")

    rows_dev = list(weight_dev.items()) + [(f"reranked: {m}", s) for m, s in model_dev.items()]
    doc = f"""# Results: weighting and reranking (step 4)

Generated by `scripts/run_rerank.py`. Earlier steps: [keyword search](results.md), [vector and hybrid search](results-vector-hybrid.md).

## Setup
- **First search:** keyword (BM25) and vector top 100 lists, fused with reciprocal rank fusion (k = 60). The vector list's weight was chosen from {", ".join(f"{w:g}" for w in WEIGHTS)} on CSF 2.0 dev questions by MRR.
- **Reranking:** the top {CANDIDATES} of the fused list are rescored by a cross-encoder that reads the question together with each control's title and statement (cut at {MAX_TOKENS} tokens). On test questions, these {CANDIDATES} candidates hold {recall_at_candidates['csf2']:.1%} of CSF 2.0's right controls and {recall_at_candidates['sp800_171']:.1%} of SP 800-171's, which is as far as reranking can go.
- **Models tried** (on this computer's CPU, at pinned versions): {"; ".join(f"{m} ({lic})" for m, (_, lic) in MODELS.items())}. The best on CSF 2.0 dev questions by MRR was then run once on the test questions.

## CSF 2.0 dev questions (used to choose)
{table(rows_dev)}

**Chosen:** vector weight {best_weight:g}; reranker {best_model}.

## Test questions
### CSF 2.0 (the hard key)
{table(list(test['csf2'].items()))}

### SP 800-171 Rev 3 (easy for keywords)
{table(list(test['sp800_171'].items()))}

## Reading these numbers
- MRR, hit@k, and recall@10 count only the top 10, as in the earlier steps.
- Every candidate's reranker score is in `results/rerank_scores.json`, so these rankings can be checked without running the models.
"""
    (ROOT / "docs" / "results-rerank.md").write_text(doc, encoding="utf-8")
    print(table(rows_dev))
    print(f"\nchosen: weight {best_weight:g}, reranker {best_model}\n")
    for name, (label, _) in KEYS.items():
        print(f"== test, {label}\n{table(list(test[name].items()))}\n")
    print(f"candidates hold (test): {recall_at_candidates} | {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
