# The MCP server's published page, setup guide, and README numbers must come from results/mcp.json and
# the code, and the results must back what the page says. The server itself is checked in test_mcp_server.py.
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from controlfinder import mcp_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "mcp.json").read_text())


def test_the_page_is_generated_from_the_results():
    spec = importlib.util.spec_from_file_location("run_mcp_check", ROOT / "scripts" / "run_mcp_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.render_doc(RESULTS) == (ROOT / "docs" / "results-mcp.md").read_text(encoding="utf-8")


def test_the_results_back_the_claims():
    assert RESULTS["tools"] == ["answer_question", "get_control", "search_controls"]
    assert RESULTS["search"]["same_top_20_as_published"] == RESULTS["search"]["questions"] == 203
    not_covered = next(a for a in RESULTS["answers"] if a["id"] == "oos-06")
    assert not_covered["covered"] is False and not_covered["answer"] == mcp_server.answer.NOT_COVERED


def test_the_setup_guide_quotes_the_servers_own_instructions():
    guide = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    assert mcp_server.INSTRUCTIONS in guide and "python -m controlfinder.mcp_server" in guide


def test_readme_numbers_match_the_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## The MCP server")
    section = readme[start:readme.index("\n## ", start)]
    s = RESULTS["search"]
    for phrase in (f"it gave the published top 20 for {s['same_top_20_as_published']} of {s['questions']} questions",
                   f"it connected in {RESULTS['connect_ms']} ms",
                   f"searches after the first took {s['later_calls_ms']['median']:.1f} ms (median)"):
        assert phrase in section, phrase
