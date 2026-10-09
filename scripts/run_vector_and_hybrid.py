# ============================================================================
# Vector search and hybrid search, scored on both answer keys.
#
# Run:   python scripts/run_vector_and_hybrid.py
#        (needs Ollama running with nomic-embed-text: ollama pull nomic-embed-text)
#   1. Embeds every active control two ways (title and statement; plus
#      guidance) and every question, with the local nomic-embed-text model.
#   2. Ranks the controls for each question three ways: keyword (BM25, with the
#      search text chosen in steps 1 and 2), vector, and hybrid (reciprocal rank
#      fusion of the keyword and vector lists, top 100 each).
#   3. Chooses the vector text and the hybrid text on CSF 2.0 dev questions only
#      (the hard key), by MRR, then scores each choice once on the test
#      questions of both keys.
#   4. Writes results/vector_hybrid.json (scores and choices),
#      results/vector_top100.json (every question's top 100 vector results, so
#      CI can check the scores without a model), and docs/results-vector-hybrid.md.
# Every score uses each method's top 10, the same as the keyword baseline.
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
from controlfinder.embeddings import DOCUMENT, MAX_TOKENS, MODEL, QUERY, cached_embed  # noqa: E402
from controlfinder.fusion import rrf  # noqa: E402
from controlfinder.metrics import summarize  # noqa: E402
from controlfinder.vector import search  # noqa: E402

DEPTH = 100
RRF_K = 60
BM25_TEXT = "title + statement + guidance"  # chosen on dev in steps 1 and 2
TEXTS = {
    "title + statement": lambda c: f"{c.title}. {c.statement}",
    "title + statement + guidance": lambda c: f"{c.title}. {c.statement} {c.guidance}",
}
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
    ids = [c.id for c in controls]
    keys = {name: load() for name, (_, load) in KEYS.items()}

    bm25 = BM25([TEXTS[BM25_TEXT](c) for c in controls])
    keyword = {name: {q.id: [ids[i] for i, _ in bm25.search(q.text, DEPTH)] for q in qs} for name, qs in keys.items()}

    query_vectors = {name: cached_embed(f"questions-{name}", [q.text for q in qs], QUERY)[0] for name, qs in keys.items()}
    vector, cut = {}, {}
    for text_name, text in TEXTS.items():
        slug = text_name.replace(" + ", "-").replace(" ", "-")
        doc_vectors, cut[text_name] = cached_embed(f"controls-{slug}", [text(c) for c in controls], DOCUMENT)
        vector[text_name] = {name: {q.id: [ids[i] for i in row] for q, row in zip(qs, search(query_vectors[name], doc_vectors, DEPTH))}
                             for name, qs in keys.items()}
    hybrid = {t: {name: {qid: rrf([keyword[name][qid], vector[t][name][qid]], k=RRF_K, depth=DEPTH)[:DEPTH]
                         for qid in keyword[name]} for name in keys} for t in TEXTS}

    csf = keys["csf2"]
    dev = {"keyword (BM25)": score(keyword["csf2"], csf, "dev")}
    dev.update({f"vector: {t}": score(vector[t]["csf2"], csf, "dev") for t in TEXTS})
    dev.update({f"hybrid: {t}": score(hybrid[t]["csf2"], csf, "dev") for t in TEXTS})
    best_vector = max(TEXTS, key=lambda t: (dev[f"vector: {t}"]["mrr"], dev[f"vector: {t}"]["recall@10"]))
    best_hybrid = max(TEXTS, key=lambda t: (dev[f"hybrid: {t}"]["mrr"], dev[f"hybrid: {t}"]["recall@10"]))
    test = {name: {"keyword (BM25)": score(keyword[name], qs, "test"),
                   "vector": score(vector[best_vector][name], qs, "test"),
                   "hybrid": score(hybrid[best_hybrid][name], qs, "test")} for name, qs in keys.items()}

    results = {
        "model": {"name": MODEL, "parameters": "137M", "license": "Apache-2.0", "dimensions": 768,
                  "max_tokens": MAX_TOKENS, "controls_at_token_limit": cut, "controls": len(controls)},
        "settings": {"bm25_text": BM25_TEXT, "rrf_k": RRF_K, "fusion_depth": DEPTH, "scored_on": "top 10",
                     "chosen_on": "CSF 2.0 dev questions, by MRR"},
        "dev_csf2": dev,
        "chosen": {"vector": best_vector, "hybrid": best_hybrid},
        "test": test,
    }
    (ROOT / "results" / "vector_hybrid.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (ROOT / "results" / "vector_top100.json").write_text(json.dumps(vector, separators=(",", ":")) + "\n", encoding="utf-8")

    doc = f"""# Results: vector search and hybrid search

Generated by `scripts/run_vector_and_hybrid.py`. The keyword baseline is in [results.md](results.md).

## Setup
- **Embedding model:** {MODEL} through Ollama, on this computer (137M parameters, Apache-2.0, 768 dimensions). Controls start with "{DOCUMENT.strip()}" and questions with "{QUERY.strip()}", the prefixes the model was trained with.
- **Token limit:** the model reads at most {MAX_TOKENS:,} tokens of a text. Of the {len(controls):,} controls, {cut['title + statement']} reached the limit as title and statement, and {cut['title + statement + guidance']} with guidance added (their ends were cut).
- **Hybrid:** reciprocal rank fusion of the keyword and vector top {DEPTH} lists, with k = {RRF_K} (the value from Cormack, Clarke, and Büttcher, 2009; not tuned).
- **Scoring:** every method's top 10, the same as the keyword baseline.
- **Choices:** the text for vector search and for hybrid search were chosen on CSF 2.0 dev questions only, by MRR. Each choice was then scored once on the test questions of both keys.

## CSF 2.0 dev questions (used to choose)
{table(list(dev.items()))}

**Chosen:** vector search on "{best_vector}"; hybrid search with vectors on "{best_hybrid}".

## Test questions
### CSF 2.0 (the hard key)
{table(list(test['csf2'].items()))}

### SP 800-171 Rev 3 (easy for keywords)
{table(list(test['sp800_171'].items()))}

## Reading these numbers
- **MRR:** 1 divided by the rank of the first right control, averaged, counting only the top 10.
- **hit@k:** how often a right control is in the top k. **recall@10:** the share of the right controls in the top 10; on CSF 2.0 it can't reach 1 for outcomes with more than 10 right controls.
- Every question's top 100 vector results are in `results/vector_top100.json`, so the vector and hybrid scores can be checked without running the model.
"""
    (ROOT / "docs" / "results-vector-hybrid.md").write_text(doc, encoding="utf-8")
    print(table(list(dev.items())))
    print(f"\nchosen: vector '{best_vector}', hybrid '{best_hybrid}'\n")
    for name, (label, _) in KEYS.items():
        print(f"== test, {label}\n{table(list(test[name].items()))}\n")
    print(f"controls at the {MAX_TOKENS} token limit: {cut} | {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
