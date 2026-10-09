# The CSF 2.0 answer key must hold NIST's 106 outcomes and only real, active SP 800-53 controls.
import importlib.util
from pathlib import Path

import pytest

from controlfinder.answer_key import CSF_KEY, load_csf_questions
from controlfinder.catalog import active_controls

ROOT = Path(__file__).resolve().parent.parent


def test_counts_and_split():
    questions = load_csf_questions()
    assert len(questions) == 106
    assert sum(len(q.expected) for q in questions) == 731
    assert (sum(q.split == "dev" for q in questions), sum(q.split == "test" for q in questions)) == (60, 46)


def test_known_outcomes():
    by_id = {q.id: q for q in load_csf_questions()}
    assert by_id["GV.OC-01"].expected == {"pm-11"}
    assert len(by_id["PR.AA-01"].expected) == 14 and {"ac-2", "ia-2", "ia-5"} <= by_id["PR.AA-01"].expected
    assert by_id["ID.RA-01"].text == "Vulnerabilities in assets are identified, validated, and recorded"
    assert "ra-5" in by_id["ID.RA-01"].expected  # Vulnerability Monitoring and Scanning


def test_every_answer_is_an_active_control_and_no_withdrawn_outcomes():
    active = {c.id for c in active_controls()}
    for q in load_csf_questions():
        assert q.expected and q.expected <= active, q.id
        assert q.text and "Withdrawn" not in q.text, q.id


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "NIST_CSF_2.0_reference_tool.xlsx").exists(),
                    reason="NIST's spreadsheet isn't downloaded (CI uses the committed key)")
def test_committed_key_matches_the_spreadsheet():
    spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build_csf_answer_key.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    import json
    committed = json.loads(CSF_KEY.read_text(encoding="utf-8"))["questions"]
    assert build.build(build.rows(build.RAW)) == committed
