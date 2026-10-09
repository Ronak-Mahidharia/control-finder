# Step 5's published page and README numbers must come from results/database.json, and the results
# must back what the page says. The database itself is checked in test_database.py.
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "database.json").read_text())


def test_the_page_is_generated_from_the_results():
    spec = importlib.util.spec_from_file_location("run_database_check", ROOT / "scripts" / "run_database_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.render_doc(RESULTS) == (ROOT / "docs" / "results-database.md").read_text(encoding="utf-8")


def test_the_results_back_the_claims():
    e, settings = RESULTS["exact"], RESULTS["hnsw"]["settings"]
    assert e["same_top_10"] == e["questions"] == 203
    assert all(e["published_test_scores_reproduced"].values())
    assert len(e["swaps"]) == e["questions"] - e["same_top_100"]
    assert all(s["similarity_gap"] < 1e-6 for s in e["swaps"])
    assert settings["ef_search 40 (default)"]["rows_returned_for_top_100"] == [40]
    assert settings["ef_search 100"]["rows_returned_for_top_100"] == [100]
    assert settings["ef_search 100"]["test_scores_that_changed"] == {}
    assert RESULTS["planner_with_the_index_not_forced"] == {"top 10": "Index Scan", "top 100": "Seq Scan"}


def test_readme_numbers_match_the_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## The vector database (step 5)"):readme.index("## Run it")]
    e, default = RESULTS["exact"], RESULTS["hnsw"]["settings"]["ef_search 40 (default)"]
    ms = {name: m["median"] for name, m in RESULTS["speed"]["milliseconds_per_question_top_100"].items()}
    changed = default["hybrid_top_10_changed"]["hybrid (step 3)"]
    hit10 = default["test_scores_that_changed"]["csf2"]["hybrid (step 3)"]["hit@10"]
    assert hit10["min"] == hit10["max"]  # the README gives one number
    for phrase in (f"for all {e['questions']} questions",
                   f"In {len(e['swaps'])} of the {e['questions']} top-100 lists",
                   f"for {changed['min']} to {changed['max']} of the {e['questions']} questions",
                   f"for {hit10['min']:.1%} of questions instead of {hit10['exact']:.1%}",
                   f"At {RESULTS['load']['controls']:,} controls",
                   f"the index took {ms['database: HNSW, ef_search 100']:.2f} ms per question",
                   f"against {ms['database: exact']:.2f} ms for exact search",
                   f"Sending the question alone took {ms['database: sending the question only (no search)']:.2f} ms",
                   f"the in-memory search took {ms['in memory (numpy, steps 3 and 4)']:.2f} ms"):
        assert phrase in section, phrase
