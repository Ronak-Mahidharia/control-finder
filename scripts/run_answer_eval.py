# ============================================================================
# Step 6: cited answers from a local model, measured.
#
# Run:   Ollama with granite4:3b, qwen3:8b, and nomic-embed-text
#        python scripts/run_answer_eval.py     (about 55 minutes on this Mac;
#        answers are cached in data/processed/, so a rerun skips them)
# Questions, each split into dev and test halves:
#   - NIST's answer keys (SP 800-171 Rev 3 and CSF 2.0): citation precision,
#     recall, and F1 against NIST's own links.
#   - Plain-English questions (hand-made): right when an anchor is cited.
#   - Questions SP 800-53 doesn't answer (hand-made): right when the answer
#     says the question isn't covered.
# Each question's shown controls are the top k of the weighted hybrid search
# (step 4's settings), computed from the committed embeddings.
# Choosing, on dev questions only (fixed before any run):
#   1. Both models with both prompt wordings, shown the top 10.
#   2. The best of those four, shown the top 5 instead.
#   "Best" is the highest mean of three dev scores: citation F1 on the NIST
#   keys, the share of plain-English questions answered with an anchor, and
#   the share of out-of-scope questions answered "not covered". A tie goes to
#   the faster setup. The best of the five then runs once on the test questions.
# Writes results/answers.json (scores and choices), results/answers_all.json
# (every answer, so CI can score them again without a model), and
# docs/results-answers.md.
# ============================================================================
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controlfinder import answer  # noqa: E402
from controlfinder.answer_key import load_csf_questions, load_out_of_scope, load_plain_questions, load_questions  # noqa: E402
from controlfinder.bm25 import BM25  # noqa: E402
from controlfinder.catalog import active_controls  # noqa: E402
from controlfinder.embeddings import committed_controls  # noqa: E402
from controlfinder.fusion import rrf  # noqa: E402
from controlfinder.search import DEPTH, RRF_K, VECTOR_WEIGHT  # noqa: E402
from controlfinder.vector import search  # noqa: E402

MODELS = ["granite4:3b", "qwen3:8b"]
VERSIONS = ["v1", "v2"]
FIRST_K, SECOND_K = 10, 5
SETS = {"csf2": load_csf_questions, "sp800_171": load_questions, "plain_english": load_plain_questions,
        "out_of_scope": load_out_of_scope}
NIST = ("csf2", "sp800_171")
CACHE = ROOT / "data" / "processed" / "answers-cache.jsonl"


def config_name(model, version, k):
    return f"{model}, prompt {version}, top {k}"


def model_details():
    with urllib.request.urlopen(answer.CHAT_URL.replace("/api/chat", "/api/tags"), timeout=30) as response:
        tags = {m["name"]: m for m in json.load(response)["models"]}
    out = {}
    for name in MODELS:
        m = tags[name]
        out[name] = {"digest": m["digest"], "parameters": m["details"]["parameter_size"],
                     "quantization": m["details"]["quantization_level"], "license": "Apache-2.0"}
    return out


def retrieval():
    """Every question's fused list (keyword top 100 and vector top 100, weighted RRF), from committed embeddings."""
    controls = active_controls()
    docs = committed_controls(controls)
    manifest = json.loads((ROOT / "embeddings" / "manifest.json").read_text())
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    fused = {}
    for key, load in SETS.items():
        questions = load()
        entry = manifest["questions"][key]
        assert entry["ids"] == [q.id for q in questions], key
        vectors = np.load(ROOT / "embeddings" / entry["file"])
        for q, row in zip(questions, search(vectors, docs, DEPTH)):
            keyword = [controls[i].id for i, _ in bm25.search(q.text, DEPTH)]
            fused[(key, q.id)] = rrf([keyword, [controls[i].id for i in row]], k=RRF_K, depth=DEPTH, weights=[1.0, VECTOR_WEIGHT])
    return fused


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            cache[entry["key"]] = entry["answer"]
    return cache


def run(config, items, by_id, fused, digests, cache):
    """The model's answer to each (set, question), from the cache when this exact request was answered before."""
    model, version, k = config
    records, started = [], time.time()
    for n, (key, q) in enumerate(items, 1):
        shown = [by_id[c] for c in fused[(key, q.id)][:k]]
        text = answer.prompt(q.text, shown, version)
        cache_key = hashlib.sha256(json.dumps([model, digests[model], answer.SYSTEM, text, answer.OPTIONS]).encode()).hexdigest()
        if cache_key not in cache:
            cache[cache_key] = answer.ask(q.text, shown, model, version)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": cache_key, "answer": cache[cache_key]}) + "\n")
        records.append({"config": config_name(*config), "set": key, "id": q.id, "split": q.split,
                        "shown": [c.id for c in shown], **cache[cache_key]})
        if n % 25 == 0 or n == len(items):
            print(f"  {config_name(*config)}: {n}/{len(items)} ({time.time() - started:.0f} s)", flush=True)
    return records


def score(records, questions, by_id):
    """The scores of one setup's answers on one split. questions: {(set, id): question}."""
    nist, plain, oos, seconds, invalid, unshown = [], [], [], [], 0, 0
    refusals = {"explains": 0, "empty": 0, "answers anyway": 0}
    by_key = {key: [] for key in NIST}
    kinds = {}
    for r in records:
        shown = [by_id[c] for c in r["shown"]]
        parsed = answer.parse(r["raw"], shown)
        seconds.append(r["seconds"])
        if parsed is None:
            invalid += 1
            parsed = (False, "", [])
        covered, text, cited = parsed
        if answer.mentioned(text) - {c.label for c in shown}:
            unshown += 1
        if not covered:
            refusals[answer.refusal_text(text)] += 1
        q = questions[(r["set"], r["id"])]
        if r["set"] in NIST:
            p, rc, f1 = answer.citation_scores(cited, q.expected)
            row = (p, rc, f1, not covered, len(cited))
            nist.append(row)
            by_key[r["set"]].append(row)
        elif r["set"] == "plain_english":
            plain.append((covered and bool(set(cited) & q.expected), not covered))
        else:
            oos.append(not covered)
            kinds.setdefault(q.kind, []).append(not covered)

    def nist_summary(rows):
        return {"questions": len(rows), "precision": round(statistics.mean(r[0] for r in rows), 3),
                "recall": round(statistics.mean(r[1] for r in rows), 3), "f1": round(statistics.mean(r[2] for r in rows), 3),
                "said_not_covered": sum(r[3] for r in rows), "citations_per_answer": round(statistics.mean(r[4] for r in rows), 2)}

    out = {"nist": {**nist_summary(nist), "by_key": {key: nist_summary(rows) for key, rows in by_key.items()}},
           "plain_english": {"questions": len(plain), "anchor_cited": round(statistics.mean(h for h, _ in plain), 3),
                             "said_not_covered": sum(n for _, n in plain)},
           "out_of_scope": {"questions": len(oos), "said_not_covered": round(statistics.mean(oos), 3),
                            "by_kind": {kind: round(statistics.mean(v), 3) for kind, v in sorted(kinds.items())}},
           "answers": {"count": len(records), "invalid_json": invalid, "mention_an_unshown_control": unshown,
                       "not_covered_text": refusals,
                       "median_seconds": round(statistics.median(seconds), 2),
                       "p95_seconds": round(sorted(seconds)[int(0.95 * (len(seconds) - 1))], 2)}}
    out["score"] = round(statistics.mean([out["nist"]["f1"], out["plain_english"]["anchor_cited"], out["out_of_scope"]["said_not_covered"]]), 4)
    return out


def details(records, questions, by_id):
    """The plain-English questions answered without an anchor, and the out-of-scope ones answered as covered."""
    misses, covered_oos = [], []
    for r in records:
        q = questions[(r["set"], r["id"])]
        covered, text, cited = answer.parse(r["raw"], [by_id[c] for c in r["shown"]]) or (False, "", [])
        if r["set"] == "plain_english" and not (covered and set(cited) & q.expected):
            misses.append({"id": q.id, "question": q.text, "anchors": sorted(by_id[a].label for a in q.expected),
                           "an_anchor_was_shown": bool(set(r["shown"]) & q.expected), "cited": [by_id[c].label for c in cited],
                           "cites_an_anchors_enhancement": any(by_id[c].parent in q.expected for c in cited)})
        elif r["set"] == "out_of_scope" and covered:
            covered_oos.append({"id": q.id, "question": q.text, "kind": q.kind, "cited": [by_id[c].label for c in cited]})
    return {"plain_english_misses": misses, "out_of_scope_answered_covered": covered_oos}


def best(dev):
    return max(dev, key=lambda name: (dev[name]["score"], -dev[name]["answers"]["median_seconds"]))


def baselines(fused, questions, split, by_id):
    """Citing the search's top results directly, with no model. It never says "not covered"."""
    out = {}
    for k in (1, 3, 5, 10):
        records = [{"set": key, "id": qid, "shown": fused[(key, qid)][:k], "seconds": 0.0,
                    "raw": json.dumps({"covered": True, "answer": "", "citations": [by_id[c].label for c in fused[(key, qid)][:k]]})}
                   for (key, qid), q in questions.items() if q.split == split]
        out[f"cite the top {k}"] = score(records, questions, by_id)
    return out


def main():
    started = time.time()
    controls = active_controls()
    by_id = {c.id: c for c in controls}
    questions = {(key, q.id): q for key, load in SETS.items() for q in load()}
    fused = retrieval()
    models = model_details()
    digests = {name: d["digest"] for name, d in models.items()}
    cache = load_cache()
    items = {split: [(key, q) for (key, _), q in questions.items() if q.split == split] for split in ("dev", "test")}

    all_records, dev = [], {}
    for model in MODELS:
        for version in VERSIONS:
            records = run((model, version, FIRST_K), items["dev"], by_id, fused, digests, cache)
            all_records += records
            dev[config_name(model, version, FIRST_K)] = score(records, questions, by_id)
    first = best(dev)
    model, version = next((m, v) for m in MODELS for v in VERSIONS if config_name(m, v, FIRST_K) == first)
    records = run((model, version, SECOND_K), items["dev"], by_id, fused, digests, cache)
    all_records += records
    dev[config_name(model, version, SECOND_K)] = score(records, questions, by_id)
    chosen = best(dev)
    model, version, k = next((m, v, kk) for m in MODELS for v in VERSIONS for kk in (FIRST_K, SECOND_K)
                             if config_name(m, v, kk) == chosen)
    test_records = run((model, version, k), items["test"], by_id, fused, digests, cache)
    all_records += test_records
    test = score(test_records, questions, by_id)

    ceiling = {}
    for kk in (SECOND_K, FIRST_K):
        rows = [len(set(fused[(key, q.id)][:kk]) & q.expected) / len(q.expected) for (key, _), q in questions.items()
                if key in NIST and q.split == "test"]
        ceiling[f"top {kk}"] = round(statistics.mean(rows), 3)
    results = {
        "models": models,
        "settings": {"prompts": answer.PROMPTS, "system": answer.SYSTEM, "options": answer.OPTIONS,
                     "statement_chars": answer.STATEMENT_CHARS, "search": {"depth": DEPTH, "rrf_k": RRF_K, "vector_weight": VECTOR_WEIGHT},
                     "chosen_on": "dev: the mean of NIST citation F1, plain-English anchor cited, and out-of-scope answered not covered"},
        "questions": {split: {key: sum(1 for (kk, _), q in questions.items() if kk == key and q.split == split) for key in SETS}
                      for split in ("dev", "test")},
        "dev": dev, "first_round_best": first, "chosen": chosen,
        "test": test, "test_details": details(test_records, questions, by_id),
        "test_baselines": baselines(fused, questions, "test", by_id),
        "test_recall_ceiling_nist": ceiling,
    }
    (ROOT / "results" / "answers.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (ROOT / "results" / "answers_all.json").write_text(json.dumps(all_records, separators=(",", ":")) + "\n", encoding="utf-8")
    doc = render_doc(results, examples(test_records + all_records, questions, by_id, chosen))
    (ROOT / "docs" / "results-answers.md").write_text(doc, encoding="utf-8")
    print(doc)
    print(f"{time.time() - started:.0f} s")


def examples(records, questions, by_id, chosen):
    """A few of the chosen setup's answers, word for word: plain-01 (the password example), then the first
    security-adjacent and the first unrelated out-of-scope test questions."""
    picks = []
    wanted = [("plain_english", "plain-01")]
    oos = [q for (key, _), q in questions.items() if key == "out_of_scope" and q.split == "test"]
    wanted += [("out_of_scope", next(q.id for q in oos if q.kind == kind)) for kind in ("security-adjacent", "unrelated")]
    for key, qid in wanted:
        r = next(r for r in records if r["config"] == chosen and r["set"] == key and r["id"] == qid)
        covered, text, cited = answer.parse(r["raw"], [by_id[c] for c in r["shown"]]) or (False, "", [])
        picks.append({"id": qid, "split": r["split"], "question": questions[(key, qid)].text, "covered": covered,
                      "answer": text, "cited": [by_id[c].label for c in cited]})
    return picks


def render_doc(r, picks):
    """docs/results-answers.md, from results/answers.json and the chosen setup's example answers."""
    def row(name, s):
        n = s["nist"]
        return (f"| {name} | {n['f1']:.3f} ({n['precision']:.3f} / {n['recall']:.3f}) | {n['citations_per_answer']:.2f} | "
                f"{s['plain_english']['anchor_cited']:.1%} | {s['out_of_scope']['said_not_covered']:.1%} | "
                f"{s['score']:.3f} | {s['answers']['median_seconds']:.2f} |")

    head = ("| Setup | NIST citation F1 (precision / recall) | Citations per answer | Plain English: an anchor cited | "
            "Out of scope: said not covered | Score | Median seconds |\n|---|---|---|---|---|---|---|")
    t = r["test"]
    nc, dnc = t["answers"]["not_covered_text"], r["dev"][r["chosen"]]["answers"]["not_covered_text"]
    refused = sum(nc.values())
    d = r["test_details"]
    covered_oos = "; ".join(f"{x['id']} (\"{x['question']}\"), citing {', '.join(x['cited'])}" for x in d["out_of_scope_answered_covered"]) or "none"
    misses = "; ".join(f"{x['id']}, which cited {', '.join(x['cited']) or 'nothing'} (anchors: {', '.join(x['anchors'])}"
                       + ("" if x["an_anchor_was_shown"] else "; no anchor was shown")
                       + ("; it cited an enhancement of an anchor" if x["cites_an_anchors_enhancement"] else "") + ")"
                       for x in d["plain_english_misses"]) or "none"
    kinds = ", ".join(f"{kind} {share:.1%}" for kind, share in t["out_of_scope"]["by_kind"].items())
    example_lines = "\n".join(
        f"- **{p['id']}** ({p['split']}): \"{p['question']}\" → {'covered' if p['covered'] else 'not covered'}"
        + (f", citing {', '.join(p['cited'])}" if p["cited"] else "") + f". The answer: \"{p['answer']}\"" for p in picks)
    models = "; ".join(f"{name} ({m['parameters']}, {m['quantization']}, {m['license']})" for name, m in r["models"].items())
    q = r["questions"]
    return f"""# Results: cited answers (step 6)

Generated by `scripts/run_answer_eval.py`. Earlier steps: [the API](results-api.md), [weighting and reranking](results-rerank.md).

## Setup
- **Models:** {models}, through Ollama on this computer, at temperature 0 with a fixed seed.
- **What the model sees:** the top 5 or 10 controls of the weighted hybrid search (step 4), each with its label, title, and statement (cut at {r['settings']['statement_chars']} characters), then the question.
- **What it returns:** JSON with "covered", an answer, and citations. The JSON schema allows only the shown controls' labels as citations, so a citation can't name a control the model wasn't shown.
- **Questions:**
  - NIST's two answer keys ({q['dev']['csf2'] + q['dev']['sp800_171']} dev and {q['test']['csf2'] + q['test']['sp800_171']} test), scored by citation precision, recall, and F1 against NIST's links.
  - Plain-English questions, hand-made ({q['dev']['plain_english']} dev and {q['test']['plain_english']} test). Right when the answer cites one of the question's anchor controls.
  - Questions SP 800-53 doesn't answer, hand-made ({q['dev']['out_of_scope']} dev and {q['test']['out_of_scope']} test), half security-adjacent and half unrelated. Right when the answer says the question isn't covered.
- **Choosing:** on dev questions only. Both models with both prompt wordings were shown the top 10, then the best of those four was shown the top 5. The score is the mean of NIST citation F1, the plain-English anchor rate, and the out-of-scope rate.

## Dev questions (used to choose)
{head}
{chr(10).join(row(name, s) for name, s in r['dev'].items())}

**Chosen:** {r['chosen']}.

## Test questions
{head}
{row(r['chosen'], t)}
{chr(10).join(row(name + ' (no model)', s) for name, s in r['test_baselines'].items())}

- **By key** (test, NIST citation F1): {'; '.join(f"{key} {s['f1']:.3f} ({s['questions']} questions)" for key, s in t['nist']['by_key'].items())}.
- **The search's ceiling:** the shown controls held {r['test_recall_ceiling_nist']['top 5']:.1%} (top 5) and {r['test_recall_ceiling_nist']['top 10']:.1%} (top 10) of NIST's linked controls on test, so citations can't reach more than that.
- **Out of scope, by kind** (test, said not covered): {kinds}.
- **Said "not covered" to a covered question** (test): {t['nist']['said_not_covered']} of {t['nist']['questions']} NIST questions and {t['plain_english']['said_not_covered']} of {t['plain_english']['questions']} plain-English ones.
- **Out of scope but answered as covered** (test): {covered_oos}.
- **Plain-English questions answered without an anchor** (test): {misses}.
- **Faithfulness:** citations can only name shown controls, and {t['answers']['mention_an_unshown_control']} of {t['answers']['count']} test answers mention a control that wasn't shown. Answers that weren't valid JSON: {t['answers']['invalid_json']}.
- **When the model says "not covered", its own text often answers anyway.** Of its {refused} "not covered" answers on test, {nc['answers anyway']} answer the question from what the model already knew (the HTTPS port, Log4Shell's CVSS score, the capital of Australia), {nc['explains']} say that no control addresses it, and {nc['empty']} are empty (on dev: {dnc['answers anyway']} answer anyway, {dnc['explains']} explain, and {dnc['empty']} are empty). That text isn't from the catalog, so the API never shows it: a "not covered" answer always gets the same fixed sentence. This check and the fixed sentence were added after reading the test answers; they change no score or choice.

## Example answers (the chosen setup, word for word)
{example_lines}

## Reading these numbers
- **Citation precision:** the share of an answer's citations that NIST links to the question. **Recall:** the share of NIST's links that the answer cites. Both are averaged over questions; an answer that cites nothing scores 0.
- **Plain-English and out-of-scope questions are hand-made**, so they show behavior, not an official benchmark. The files say how each was made (`answer_keys/plain_english.json`, `answer_keys/out_of_scope.json`).
- **No model** rows cite the search's top results directly: they never say "not covered", so they score 0 on out-of-scope questions.
- Every answer is in `results/answers_all.json`, so these scores can be checked without running the models.
"""


if __name__ == "__main__":
    main()
