# The AI judge and the question rewriting: the published numbers must come back from committed data alone (the kept
# answers and judge replies, the kept rewrites and their vector results, and NIST's catalog), and the README must match.
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_judge, run_rewrite = script("run_judge"), script("run_rewrite")
JUDGED = json.loads((ROOT / "results" / "judge.json").read_text(encoding="utf-8"))
REWRITE = json.loads((ROOT / "results" / "rewrite.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_judges_replies_are_for_step_6s_chosen_test_answers_only():
    rows = run_judge.judged_answers()
    assert len(rows) == JUDGED["answers"] == 130 and {r["split"] for r in rows} == {"test"}
    replies = json.loads(run_judge.KEPT.read_text(encoding="utf-8"))
    assert sorted((r["judge"], r["set"], r["id"]) for r in replies) == sorted(
        (m, r["set"], r["id"]) for m in run_judge.JUDGES for r in rows)


def test_every_judge_score_comes_back():
    by_id = {c.id: c for c in run_judge.active_controls()}
    questions = {(key, q.id): q for key, (_, load) in run_judge.SETS.items() for q in load()}
    replies = json.loads(run_judge.KEPT.read_text(encoding="utf-8"))
    assert run_judge.score(run_judge.judged_answers(), replies, questions, by_id) == JUDGED["judges"]


def test_every_rewriting_score_and_the_choice_come_back_without_a_model():
    kept = json.loads(run_rewrite.KEPT.read_text(encoding="utf-8"))
    results = run_rewrite.results_from(kept, live=False)
    assert json.loads(json.dumps(results)) == REWRITE
    base = REWRITE["test"]["csf2"][run_rewrite.BASE]
    step4 = json.loads((ROOT / "results" / "rerank.json").read_text())["test"]
    assert base == step4["csf2"]["hybrid, vector weight 3"]  # the base is step 4's weighted hybrid search


def test_the_readme_numbers_match():
    rows = [line.split(" | ") for line in README.splitlines() if line.startswith("| ")]
    chosen = f"with rewriting: {run_rewrite.name(REWRITE['chosen'])}"
    for key, label in (("csf2", "CSF 2.0"), ("sp800_171", "SP 800-171")):
        m = REWRITE["test"][key][chosen]
        row = next(r for r in rows if r[0] == f"| {label}" and "rewritten" in r[1])
        assert row[3:6] == [f"{m['mrr']:.3f}", f"{m['hit@1'] * 100:.1f}%", f"{m['hit@10'] * 100:.1f}% |"], row
    for model, r in JUDGED["judges"].items():
        a = r["all"]
        row = next(x for x in rows if x[0] == f"| {model}" and len(x) > 4)
        assert row[1:5] == [f"{a['agreement'] * 100:.1f}%", f"{a['kappa']:.3f}",
                            f"{round(a['wrong_answers_caught'] * a['key_says_wrong'])} of {a['key_says_wrong']}",
                            f"{round(a['flags_that_are_wrong_answers'] * a['judge_says_wrong'])} of {a['judge_says_wrong']} |"], row
