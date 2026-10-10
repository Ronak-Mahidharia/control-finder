# The API's published page and README numbers must come from results/api.json, and the results must
# back what the page says. The API itself is checked in test_api.py and test_api_database.py.
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "api.json").read_text())


def test_the_page_is_generated_from_the_results():
    spec = importlib.util.spec_from_file_location("run_api_check", ROOT / "scripts" / "run_api_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.render_doc(RESULTS) == (ROOT / "docs" / "results-api.md").read_text(encoding="utf-8")


def test_the_results_back_the_claims():
    n = RESULTS["questions"]
    assert n == 203
    assert all(count == n for counts in RESULTS["same_as_published"].values() for count in counts.values())
    assert set(RESULTS["same_as_published"]) == {"keyword", "vector", "hybrid"} and all(RESULTS["test_scores_reproduced"].values())
    assert RESULTS["server"]["settings"] == {"depth": 100, "rrf_k": 60, "vector_weight": 3.0}


def test_readme_numbers_match_the_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## The API")
    section = readme[start:readme.index("\n## ", start)]  # up to the next heading
    ms = RESULTS["milliseconds"]
    hybrid_new, hybrid_again = ms["hybrid"]["first time"], ms["hybrid"]["asked before"]
    for phrase in (f"for all {RESULTS['questions']} questions",
                   f"took {hybrid_new['request']['median']:.2f} ms (median) when the question was new",
                   f"{hybrid_new['server parts (median)']['embed']:.2f} ms of it embedding the question",
                   f"{hybrid_again['request']['median']:.2f} ms when it had been asked before",
                   f"Keyword search took {ms['keyword']['first time']['request']['median']:.2f} ms"):
        assert phrase in section, phrase
