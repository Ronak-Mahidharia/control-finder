# Following NIST's links between controls (src/controlfinder/related.py): the published numbers must come back
# from committed data alone (the keyword search recomputed, the step 3 vector lists, and NIST's catalog).
import json
from pathlib import Path

import pytest

from controlfinder.answer_key import load_csf_questions, load_questions
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.fusion import rrf
from controlfinder.metrics import summarize
from controlfinder.related import expand, fusion_scores, related_links

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "related.json").read_text())
VECTOR = json.loads((ROOT / "results" / "vector_top100.json").read_text())["title + statement + guidance"]
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}
BASE = "hybrid, vector weight 3 (step 4)"


def name(setting):
    return (f"{'both ways' if setting['both_ways'] else 'one way'}, {setting['seeds']} seeds, "
            f"weight {setting['weight']:g}")


def score(rankings, questions, split):
    return summarize([(rankings[q.id][:10], q.expected) for q in questions if q.split == split])


@pytest.fixture(scope="module")
def built():
    controls = active_controls()
    bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in controls])
    keyword = {key: {q.id: [controls[i].id for i, _ in bm25.search(q.text, 100)] for q in load()}
               for key, load in KEYS.items()}
    ids = [c.id for c in controls]
    return keyword, {False: related_links(ids), True: related_links(ids, both_ways=True)}


def ranked(keyword, links, key, setting):
    out = {}
    for qid, words in keyword[key].items():
        scores, best = fusion_scores([words, VECTOR[key][qid]], k=60, depth=100, weights=[1.0, 3.0])
        out[qid] = (expand(scores, best, links[False], 0, 0.0) if setting is None else
                    expand(scores, best, links[setting["both_ways"]], setting["seeds"], setting["weight"]))
    return out


def test_the_links_come_from_nists_catalog(built):
    _, links = built
    assert {"ia-1", "pm-9", "pm-24"} <= links[False]["ac-1"]  # NIST's own example: AC-1 lists these as related
    assert sum(len(v) for v in links[False].values()) == RESULTS["links_between_active_controls"]["one_way"]
    active = set(links[False])
    assert all(target in active and target != cid for cid, targets in links[False].items() for target in targets)
    assert all(cid in links[True][target] for cid, targets in links[False].items() for target in targets)


def test_with_no_links_the_list_is_the_published_fusion(built):
    keyword, links = built
    for key in KEYS:
        for qid, words in keyword[key].items():
            scores, best = fusion_scores([words, VECTOR[key][qid]], k=60, depth=100, weights=[1.0, 3.0])
            assert expand(scores, best, links[False], 0, 0.0) == rrf([words, VECTOR[key][qid]], k=60, depth=100,
                                                                     weights=[1.0, 3.0])


def test_a_linked_control_rises_by_its_share_of_the_seeds_score():
    scores, best = {"a": 0.05, "b": 0.04, "c": 0.01}, {"a": 1, "b": 2, "c": 3}
    assert expand(scores, best, {"a": {"c"}}, seeds=1, weight=1.0) == ["c", "a", "b"]  # 0.01 + 0.05 rises to 0.06
    assert expand(scores, best, {"a": {"c"}}, seeds=1, weight=0.25) == ["a", "b", "c"]  # 0.0225 stays last
    assert expand(scores, best, {"b": {"c"}}, seeds=1, weight=1.0) == ["a", "b", "c"]  # b isn't a seed


def test_the_setting_was_chosen_on_csf_dev(built):
    keyword, links = built
    csf = load_csf_questions()
    assert score(ranked(keyword, links, "csf2", None), csf, "dev") == RESULTS["dev_csf2"][BASE]
    for setting in RESULTS["settings"]:
        assert score(ranked(keyword, links, "csf2", setting), csf, "dev") == RESULTS["dev_csf2"][name(setting)]
    dev = RESULTS["dev_csf2"]
    best = max(RESULTS["settings"], key=lambda s: (dev[name(s)]["mrr"], dev[name(s)]["recall@10"], -s["weight"],
                                                   -s["seeds"], -s["both_ways"]))
    assert best == RESULTS["chosen"]


def test_the_test_scores_reproduce_and_the_base_is_step_4s(built):
    keyword, links = built
    step4 = json.loads((ROOT / "results" / "rerank.json").read_text())["test"]
    for key, load in KEYS.items():
        questions = load()
        assert score(ranked(keyword, links, key, None), questions, "test") == RESULTS["test"][key][BASE]
        assert RESULTS["test"][key][BASE] == step4[key]["hybrid, vector weight 3"]
        chosen = f"with links: {name(RESULTS['chosen'])}"
        assert score(ranked(keyword, links, key, RESULTS["chosen"]), questions, "test") == RESULTS["test"][key][chosen]


def test_the_readme_numbers_match():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    chosen = f"with links: {name(RESULTS['chosen'])}"
    for key, label in (("csf2", "CSF 2.0"), ("sp800_171", "SP 800-171")):
        m = RESULTS["test"][key][chosen]
        row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |") and "NIST's links" in line)
        assert f"| {m['mrr']:.3f} |" in row and f"| {m['hit@1'] * 100:.1f}% |" in row and \
            f"| {m['hit@10'] * 100:.1f}% |" in row, row
