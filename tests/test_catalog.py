# The catalog parser must match NIST's published counts and print blanks the way NIST does.
from controlfinder.catalog import active_controls, families, load_catalog


def test_counts_match_nist():
    active = active_controls()
    assert len(families()) == 20
    assert len(load_catalog()) == 1196  # 324 controls + 872 enhancements, withdrawn ones included
    assert sum(1 for c in active if c.parent is None) == 300
    assert sum(1 for c in active if c.parent is not None) == 714
    assert sum(c.objective_parts for c in active) == 3715  # SP 800-53A Rev 5.2.0 assessment objectives


def test_password_enhancement_reads_like_nist():
    ia51 = next(c for c in active_controls() if c.id == "ia-5.1")
    assert (ia51.label, ia51.title, ia51.parent) == ("IA-5(1)", "Password-based Authentication", "ia-5")
    assert ia51.family == "Identification and Authentication"
    assert "approved salted key derivation function" in ia51.statement
    assert "[Assignment: organization-defined frequency]" in ia51.statement
    assert len(ia51.objectives) == 8


def test_selection_blanks_list_their_choices():
    ac1 = next(c for c in active_controls() if c.id == "ac-1")
    assert "[Selection (one or more): organization-level; mission/business process-level; system-level]" in ac1.statement


def test_no_blank_left_unprinted_and_ids_unique():
    active = active_controls()
    assert not [c.id for c in active if "{{" in c.statement + c.guidance + " ".join(c.objectives)]
    assert len({c.id for c in active}) == len(active)
    ids = {c.id for c in load_catalog()}
    assert all(c.parent in ids for c in active if c.parent)
