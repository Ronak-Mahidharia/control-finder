# The write-up's numbers must be the published ones: every phrase below is built from the results files,
# so a number that changes there fails here until the write-up is updated. The README's and the
# write-up's test counts must match the tests that exist.
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "docs" / "write-up.md").read_text(encoding="utf-8")


def results(name):
    return json.loads((ROOT / "results" / name).read_text())


def pct(x):
    return f"{x:.1%}"


def assert_all_in(phrases):
    for phrase in phrases:
        assert phrase in TEXT, phrase


def test_the_answer_key_and_search_numbers():
    sp, csf = results("keyword_baseline.json")["test"], results("keyword_baseline_csf2.json")["test"]
    vh, rr = results("vector_hybrid.json")["test"], results("rerank.json")
    weighted, reranked = "hybrid, vector weight 3", "reranked: mixedbread-ai/mxbai-rerank-xsmall-v1"
    rows = {"Keyword (BM25)": (vh["csf2"]["keyword (BM25)"], vh["sp800_171"]["keyword (BM25)"]),
            "Vector (nomic-embed-text, local)": (vh["csf2"]["vector"], vh["sp800_171"]["vector"]),
            "Hybrid (reciprocal rank fusion)": (vh["csf2"]["hybrid"], vh["sp800_171"]["hybrid"]),
            "Hybrid, the vector list weighted 3 times": (rr["test"]["csf2"][weighted], rr["test"]["sp800_171"][weighted])}
    lines = [f"| {name} | {c['mrr']:.3f} | {pct(c['hit@10'])} | {s['mrr']:.3f} |" for name, (c, s) in rows.items()]
    c, s = rr["test"]["csf2"][reranked], rr["test"]["sp800_171"][reranked]
    lines.append(f"| Weighted hybrid, top 30 reranked | **{c['mrr']:.3f}** | {pct(c['hit@10'])} | {s['mrr']:.3f} |")
    dev = rr["dev_csf2"]
    assert_all_in(lines + [
        f"BM25 scored an MRR of {sp['mrr']:.3f} on the test questions and put a right control first for {pct(sp['hit@1'])}",
        f"Keyword search dropped to an MRR of {csf['mrr']:.3f}. It put a right control first for {pct(csf['hit@1'])}",
        f"in the top 10 for {pct(csf['hit@10'])}",
        f"(recall@10 {vh['csf2']['hybrid']['recall@10']:.3f}, against {vh['csf2']['vector']['recall@10']:.3f} for vectors alone)",
        f"mxbai-rerank-xsmall-v1, scored best (MRR {dev['rerankers']['reranked: mixedbread-ai/mxbai-rerank-xsmall-v1']['mrr']:.3f})",
        f"({dev['rerankers']['reranked: BAAI/bge-reranker-base']['mrr']:.3f} against {dev['weights'][weighted]['mrr']:.3f})",
        f"held {pct(rr['test_recall_of_candidates']['csf2'])} of CSF 2.0's right controls"])


def test_the_database_and_api_numbers():
    db, api = results("database.json"), results("api.json")
    e, default = db["exact"], db["hnsw"]["settings"]["ef_search 40 (default)"]
    changed = default["hybrid_top_10_changed"]["hybrid (step 3)"]
    ms = {name: m["median"] for name, m in db["speed"]["milliseconds_per_question_top_100"].items()}
    hybrid = api["milliseconds"]["hybrid"]
    assert e["same_top_10"] == e["questions"] and all(v == api["questions"] for c in api["same_as_published"].values() for v in c.values())
    assert_all_in([
        f"It gave the same top 10 for all {e['questions']} questions",
        f"In {len(e['swaps'])} of the {e['questions']} top-100 lists",
        f"Its HNSW index keeps {default['rows_returned_for_top_100'][0]} candidates by default",
        f"for {changed['min']} to {changed['max']} of the {e['questions']} questions",
        f"Exact search took {ms['database: exact']:.2f} ms per question (median) and the forced index {ms['database: HNSW, ef_search 100']:.2f} ms",
        f"in memory took {ms['in memory (numpy, steps 3 and 4)']:.2f} ms. Sending the question alone took "
        f"{ms['database: sending the question only (no search)']:.2f} ms",
        f"A hybrid search took {hybrid['first time']['request']['median']:.2f} ms when the question was new, "
        f"{hybrid['first time']['server parts (median)']['embed']:.2f} ms of it embedding the question, and "
        f"{hybrid['asked before']['request']['median']:.2f} ms when it had been asked before"])


def test_the_answer_and_mcp_numbers():
    a, mcp = results("answers.json"), results("mcp.json")
    t, n = a["test"], a["test"]["nist"]
    granite = sorted(s["out_of_scope"]["said_not_covered"] for name, s in a["dev"].items() if name.startswith("granite4:3b"))
    assert a["chosen"] == "qwen3:8b, prompt v1, top 10"
    oos, plain = t["out_of_scope"], t["plain_english"]
    refused = round(oos["said_not_covered"] * oos["questions"])
    nc = t["answers"]["not_covered_text"]
    assert_all_in([
        f"against {granite[0]:.0%} to {granite[-1]:.0%} for granite4:3b",
        f"F1 {n['f1']:.3f} (precision {n['precision']:.3f}, recall {n['recall']:.3f}) on NIST's keys, against "
        f"{a['test_baselines']['cite the top 1']['nist']['f1']:.3f} for citing the search's top result",
        f"SP 800-171 ({n['by_key']['sp800_171']['f1']:.3f}) than on CSF 2.0 ({n['by_key']['csf2']['f1']:.3f})",
        f"for {refused} of {oos['questions']} out-of-scope questions, and for none of the {n['questions'] + plain['questions']} covered ones",
        f"it cited an anchor for {pct(plain['anchor_cited'])}. All {len(a['test_details']['plain_english_misses'])} misses",
        f"That happened in {nc['answers anyway']} of its {sum(nc.values())} \"not covered\" answers on test",
        f"{n['citations_per_answer']:.2f} controls per answer on NIST's test questions",
        f"it gave the published top 20 for all {mcp['search']['questions']} questions, connected in {mcp['connect_ms']} ms, "
        f"and answered later searches in {mcp['search']['later_calls_ms']['median']:.1f} ms"])
    assert n["said_not_covered"] == plain["said_not_covered"] == 0 and nc["answers anyway"] == 9
    assert n["precision"] < 0.5  # "kept precision below half"


def test_the_test_counts_are_current():
    count = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M)) for p in (ROOT / "tests").glob("test_*.py"))
    assert f"{count} Python tests run on every change" in TEXT
    assert f"# {count} tests, including" in (ROOT / "README.md").read_text(encoding="utf-8")
