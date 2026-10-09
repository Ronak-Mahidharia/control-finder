# Step 6's published scores must come back from the committed answers alone (CI has no model): the
# controls each question was shown, every setup's dev scores and the choice made from them, the test
# scores and baselines, and the page and README numbers.
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "answers.json").read_text())
RECORDS = json.loads((ROOT / "results" / "answers_all.json").read_text())


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("run_answer_eval", ROOT / "scripts" / "run_answer_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def context(ev):
    by_id = {c.id: c for c in ev.active_controls()}
    questions = {(key, q.id): q for key, load in ev.SETS.items() for q in load()}
    return by_id, questions, ev.retrieval()


def test_each_question_was_shown_the_searchs_top_results(context):
    _, _, fused = context
    for r in RECORDS:
        k = len(r["shown"])
        assert r["config"].endswith(f"top {k}") and r["shown"] == fused[(r["set"], r["id"])][:k], (r["config"], r["id"])


def test_dev_scores_and_the_choice_come_back(ev, context):
    by_id, questions, _ = context
    dev = {}
    for name in RESULTS["dev"]:
        records = [r for r in RECORDS if r["config"] == name and r["split"] == "dev"]
        assert len(records) == sum(RESULTS["questions"]["dev"].values()), name
        dev[name] = ev.score(records, questions, by_id)
        assert dev[name] == RESULTS["dev"][name], name
    first_round = {name: s for name, s in dev.items() if name.endswith(f"top {ev.FIRST_K}")}
    assert len(first_round) == len(ev.MODELS) * len(ev.VERSIONS) and ev.best(first_round) == RESULTS["first_round_best"]
    assert len(dev) == len(first_round) + 1
    assert RESULTS["first_round_best"].replace(f"top {ev.FIRST_K}", f"top {ev.SECOND_K}") in dev
    assert ev.best(dev) == RESULTS["chosen"]


def test_the_test_questions_were_answered_once_with_the_chosen_setup(ev, context):
    by_id, questions, fused = context
    records = [r for r in RECORDS if r["split"] == "test"]
    assert {r["config"] for r in records} == {RESULTS["chosen"]}
    assert len(records) == len({(r["set"], r["id"]) for r in records}) == sum(RESULTS["questions"]["test"].values())
    assert ev.score(records, questions, by_id) == RESULTS["test"]
    assert ev.details(records, questions, by_id) == RESULTS["test_details"]
    assert ev.baselines(fused, questions, "test", by_id) == RESULTS["test_baselines"]


def test_the_page_is_generated_from_the_results(ev, context):
    by_id, questions, _ = context
    picks = ev.examples(RECORDS, questions, by_id, RESULTS["chosen"])
    assert ev.render_doc(RESULTS, picks) == (ROOT / "docs" / "results-answers.md").read_text(encoding="utf-8")


def test_readme_numbers_match_the_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Cited answers (step 6)")
    section = readme[start:readme.index("\n## ", start)]
    t, n = RESULTS["test"], RESULTS["test"]["nist"]
    nc = t["answers"]["not_covered_text"]
    for phrase in (f"its citations scored F1 {n['f1']:.3f} (precision {n['precision']:.3f}, recall {n['recall']:.3f})",
                   f"against {RESULTS['test_baselines']['cite the top 1']['nist']['f1']:.3f} for citing the search's top result",
                   f"The top 10 held {RESULTS['test_recall_ceiling_nist']['top 10']:.1%} of NIST's links",
                   f"SP 800-171 (F1 {n['by_key']['sp800_171']['f1']:.3f}) than on CSF 2.0 ({n['by_key']['csf2']['f1']:.3f})",
                   f"{t['out_of_scope']['said_not_covered']:.1%} of the hand-made out-of-scope test questions",
                   f"In {nc['answers anyway']} of its {sum(nc.values())} \"not covered\" answers on test",
                   f"it cited an anchor control for {t['plain_english']['anchor_cited']:.1%}",
                   f"an answer took {t['answers']['median_seconds']:.2f} s (median, test)"):
        assert phrase in section, phrase


def test_the_code_still_asks_the_way_the_published_answers_were_asked():
    from controlfinder import answer
    from controlfinder.search import DEPTH, RRF_K, VECTOR_WEIGHT

    s = RESULTS["settings"]
    assert (s["prompts"], s["system"], s["options"], s["statement_chars"]) == \
        (answer.PROMPTS, answer.SYSTEM, answer.OPTIONS, answer.STATEMENT_CHARS)
    assert s["search"] == {"depth": DEPTH, "rrf_k": RRF_K, "vector_weight": VECTOR_WEIGHT}
