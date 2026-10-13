# ============================================================================
# Searching with the question in NIST's words (src/controlfinder/rewrite.py).
#
# Run:   Ollama with qwen3:8b and nomic-embed-text
#        python scripts/run_rewrite.py            (about 15 minutes on this Mac;
#                                                  every reply and search is kept, so a rerun skips it)
#        python scripts/run_rewrite.py --report   (no model: the scores from what's kept)
#   1. The weighted hybrid search of step 4, recomputed: keyword (BM25) and the
#      committed vector top 100, fused with the vector list counting 3 times.
#   2. qwen3:8b (step 6's model) rewrites each question as 3 search queries;
#      then, reading the titles of the top 5 results of the 3-query search, it
#      writes 1 more (search again). Each query is embedded with
#      nomic-embed-text and searched by keyword and by meaning.
#   3. 8 settings, tried on CSF 2.0 dev questions only: the first 1 or all 3
#      queries; each query's lists counting 0.5 or 1 times the question's own;
#      with or without search again. The best by MRR (then recall@10, then
#      fewer queries, then no search again, then the smaller weight) runs once
#      on both keys' test questions.
# Writes results/rewrite.json (scores), results/rewrite_answers.json (every
# reply and every query's vector top 100, so CI scores again without a model),
# and docs/results-rewrite.md. Every score uses the top 10, as in steps 1 to 4.
# ============================================================================
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder import rewrite  # noqa: E402
from controlfinder.answer_key import load_csf_questions, load_questions  # noqa: E402
from controlfinder.bm25 import BM25  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402
from controlfinder.metrics import summarize  # noqa: E402
from controlfinder.search import DEPTH, RRF_K, VECTOR_WEIGHT  # noqa: E402

MODEL = "qwen3:8b"
VECTOR_TEXT = "title + statement + guidance"  # step 3's choice
SETTINGS = [{"queries": n, "weight": w, "again": a} for n in (1, 3) for w in (0.5, 1.0) for a in (False, True)]
FIRST_ROUND = {"queries": 3, "weight": 1.0, "again": False}  # what search again reads
KEYS = {"csf2": ("CSF 2.0", load_csf_questions), "sp800_171": ("SP 800-171 Rev 3", load_questions)}
METRICS = ["mrr", "hit@1", "hit@5", "hit@10", "recall@10", "precision@5"]
KEPT = ROOT / "results" / "rewrite_answers.json"
BASE = "hybrid, vector weight 3 (step 4)"


def name(s):
    return (f"{s['queries']} quer{'y' if s['queries'] == 1 else 'ies'}, weight {s['weight']:g}"
            + (", search again" if s["again"] else ""))


class Searcher:
    """Keyword and vector top 100 for any text: BM25 always, vectors from what's kept or, if allowed, embedded now."""

    def __init__(self, kept, live):
        self.controls = active_controls()
        self.by_id = {c.id: c for c in self.controls}
        self.bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in self.controls])
        self.kept, self.live, self.docs = kept, live, None

    def lists(self, text):
        keyword = [self.controls[i].id for i, _ in self.bm25.search(text, DEPTH)]
        if text not in self.kept["vectors"]:
            if not self.live:
                raise SystemExit(f"No kept vector search for {text!r}: run without --report (needs Ollama).")
            from controlfinder.embeddings import QUERY, committed_controls, embed
            from controlfinder.vector import search
            if self.docs is None:
                self.docs = committed_controls(self.controls)
            vectors, _ = embed([text], QUERY)
            self.kept["vectors"][text] = [self.controls[i].id for i in search(vectors, self.docs, DEPTH)[0]]
        return keyword, self.kept["vectors"][text]


def replies_for(kept, key, q, live):
    """The question's kept replies, asking for its 3 rewrites first if they aren't kept (and that's allowed)."""
    entry = kept["replies"].setdefault(f"{key}/{q.id}", {})
    if "rewrite" not in entry:
        if not live:
            raise SystemExit(f"No kept rewrite for {key} {q.id}: run without --report (needs Ollama).")
        entry["rewrite"] = rewrite.ask(q.text, MODEL)
    return entry


def ranked(kept, key, q, setting, base, searcher, live):
    entry = replies_for(kept, key, q, live)
    extra = [searcher.lists(t) for t in rewrite.queries(entry["rewrite"]["raw"])[:setting["queries"]]]
    if setting["again"]:
        if "again" not in entry:
            first = ranked(kept, key, q, FIRST_ROUND, base, searcher, live)
            titles = [searcher.by_id[c].title for c in first[:rewrite.AGAIN_FROM]]
            if not live:
                raise SystemExit(f"No kept search-again reply for {key} {q.id}: run without --report (needs Ollama).")
            entry["again"] = rewrite.ask(q.text, MODEL, titles) | {"read": first[:rewrite.AGAIN_FROM]}
        extra += [searcher.lists(t) for t in rewrite.queries(entry["again"]["raw"])]
    return rewrite.fuse(base, extra, setting["weight"], VECTOR_WEIGHT, k=RRF_K, depth=DEPTH)


def score(rankings, questions, split):
    return summarize([(rankings[q.id][:10], q.expected) for q in questions if q.split == split])


def table(rows):
    lines = ["| Method | Questions | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for label, m in rows:
        lines.append(f"| {label} | {m['questions']} | " + " | ".join(f"{m[x]:.3f}" for x in METRICS) + " |")
    return "\n".join(lines)


def results_from(kept, live=False):
    """Every score: the 8 settings on CSF 2.0 dev, the choice, and the chosen one on both keys' test questions. With
    live, a reply or a search that isn't kept is asked for (Ollama), and kept."""
    searcher = Searcher(kept, live)
    committed = json.loads((ROOT / "results" / "vector_top100.json").read_text())[VECTOR_TEXT]
    keys = {key: load() for key, (_, load) in KEYS.items()}

    def base(key, q):
        keyword = [searcher.controls[i].id for i, _ in searcher.bm25.search(q.text, DEPTH)]
        return keyword, committed[key][q.id]

    def rankings(key, split, setting):
        out = {}
        for q in keys[key]:
            if q.split != split:
                continue
            b = base(key, q)
            out[q.id] = (rewrite.fuse(b, [], 1.0, VECTOR_WEIGHT, k=RRF_K, depth=DEPTH) if setting is None
                         else ranked(kept, key, q, setting, b, searcher, live))
            if live:
                KEPT.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
        return out

    csf = keys["csf2"]
    dev = {BASE: score(rankings("csf2", "dev", None), csf, "dev")}
    for s in SETTINGS:
        dev[name(s)] = score(rankings("csf2", "dev", s), csf, "dev")
    best = max(SETTINGS, key=lambda s: (dev[name(s)]["mrr"], dev[name(s)]["recall@10"], -s["queries"], -s["again"],
                                        -s["weight"]))
    test = {key: {BASE: score(rankings(key, "test", None), keys[key], "test"),
                  f"with rewriting: {name(best)}": score(rankings(key, "test", best), keys[key], "test")} for key in keys}
    seconds = [e[part]["seconds"] for e in kept["replies"].values() for part in ("rewrite", "again") if part in e]
    return {"model": MODEL, "settings": SETTINGS, "first_round_for_search_again": FIRST_ROUND, "dev_csf2": dev,
            "chosen": best, "test": test,
            "replies": {"count": len(seconds), "median_seconds": round(statistics.median(seconds), 2) if seconds else None}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    live = "--report" not in argv
    kept = json.loads(KEPT.read_text(encoding="utf-8")) if KEPT.exists() else {"model": MODEL, "replies": {}, "vectors": {}}
    results = results_from(kept, live)
    dev, best, test = results["dev_csf2"], results["chosen"], results["test"]
    (ROOT / "results" / "rewrite.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["# Results: searching with the question in NIST's words", "",
             "Generated by `scripts/run_rewrite.py`. Don't edit by hand.", "",
             f"{MODEL} rewrites each question as 3 short search queries in SP 800-53's words, and can search again with "
             "1 more query after reading the titles of the top 5 results (`src/controlfinder/rewrite.py`). Each query "
             "is searched by keyword and by meaning (nomic-embed-text), and every list is fused with the question's own "
             "(weighted reciprocal rank fusion, as in step 4). Every reply and every query's vector results are kept "
             "in `results/rewrite_answers.json`, so the scores are recomputed without a model.", "",
             "## Chosen on CSF 2.0 dev questions", "", table(list(dev.items())), "",
             f"Chosen by MRR (then recall@10, then fewer queries, no search again, the smaller weight): **{name(best)}**.",
             "", "## Test questions, run once", ""]
    for key, (label, _) in KEYS.items():
        lines += [f"### {label}", "", table(list(test[key].items())), ""]
    lines += [f"Replies: {results['replies']['count']}, median {results['replies']['median_seconds']} seconds each.", ""]
    doc = "\n".join(lines)
    (ROOT / "docs" / "results-rewrite.md").write_text(doc, encoding="utf-8")
    print(doc)


if __name__ == "__main__":
    main()
