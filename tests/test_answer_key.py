# The answer key must hold every active SP 800-171 Rev 3 requirement and only real SP 800-53 controls.
from controlfinder.answer_key import load_questions, split_of, to_control_id
from controlfinder.catalog import active_controls


def test_label_conversion():
    assert to_control_id("AC-02(03)") == "ac-2.3"
    assert to_control_id("SC-07") == "sc-7"
    assert to_control_id("IR 7874") is None  # a publication, not a control


def test_counts_and_split():
    questions = load_questions()
    assert len(questions) == 97
    assert sum(len(q.expected) for q in questions) == 157
    assert (sum(q.split == "dev" for q in questions), sum(q.split == "test" for q in questions)) == (52, 45)
    assert split_of("03.01.01") == split_of("03.01.01")  # the split never changes


def test_account_management_sources():
    q = next(q for q in load_questions() if q.id == "03.01.01")
    assert q.title == "Account Management"
    assert q.expected == {"ac-2", "ac-2.3", "ac-2.5", "ac-2.13"}
    assert q.text.startswith("Define the types of system accounts allowed and prohibited.")


def test_every_answer_is_an_active_control_and_questions_read_cleanly():
    active = {c.id for c in active_controls()}
    for q in load_questions():
        assert q.expected <= active, q.id
        assert q.text and "{{" not in q.text and "SR-03" not in q.text, q.id
        assert not q.text.startswith(q.title), q.id  # the title isn't added to the question
