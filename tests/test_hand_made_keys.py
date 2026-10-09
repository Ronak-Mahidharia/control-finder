# The hand-made question sets (step 6) must be well formed: unique ids and texts, the split from each
# id's hash, anchors that are active controls, both kinds of out-of-scope question in equal numbers,
# and files that say they're hand-made.
import json

from controlfinder.answer_key import (OUT_OF_SCOPE_KEY, PLAIN_KEY, load_csf_questions, load_out_of_scope,
                                      load_plain_questions, load_questions, split_of)
from controlfinder.catalog import active_controls


def test_plain_english_questions():
    questions = load_plain_questions()
    active = {c.id for c in active_controls()}
    assert len(questions) == 32 and len({q.id for q in questions}) == len({q.text for q in questions}) == 32
    for q in questions:
        assert q.expected and q.expected <= active, q.id
        assert q.split == split_of(q.id) and q.text.endswith("?"), q.id


def test_out_of_scope_questions():
    questions = load_out_of_scope()
    assert len(questions) == 40 and len({q.id for q in questions}) == len({q.text for q in questions}) == 40
    assert sum(q.kind == "security-adjacent" for q in questions) == sum(q.kind == "unrelated" for q in questions) == 20
    assert all(q.split == split_of(q.id) and q.text.endswith("?") for q in questions)


def test_the_files_say_they_are_hand_made():
    for path in (PLAIN_KEY, OUT_OF_SCOPE_KEY):
        assert json.loads(path.read_text(encoding="utf-8"))["source"].startswith("Hand-made"), path.name


def test_no_question_is_in_two_sets():
    texts = [q.text for load in (load_questions, load_csf_questions, load_plain_questions, load_out_of_scope) for q in load()]
    assert len(texts) == len(set(texts))
