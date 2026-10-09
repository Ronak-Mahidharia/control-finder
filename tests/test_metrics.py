# Retrieval metrics on one small example, worked out by hand.
from controlfinder.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank, summarize

RANKED = ["ac-2", "ia-5", "ac-2.3", "sc-7"]
RIGHT = {"ac-2.3", "ac-2.13"}


def test_single_question():
    assert recall_at_k(RANKED, RIGHT, 1) == 0.0
    assert recall_at_k(RANKED, RIGHT, 3) == 0.5  # 1 of the 2 right controls
    assert precision_at_k(RANKED, RIGHT, 4) == 0.25
    assert hit_at_k(RANKED, RIGHT, 3) == 1.0
    assert reciprocal_rank(RANKED, RIGHT) == 1 / 3  # the first right control is third
    assert reciprocal_rank(["sc-7"], RIGHT) == 0.0


def test_summary_averages():
    summary = summarize([(RANKED, RIGHT), (["ac-2.13"], RIGHT)])
    assert summary["questions"] == 2
    assert summary["mrr"] == round((1 / 3 + 1) / 2, 3)
    assert summary["hit@1"] == 0.5
    assert summary["recall@5"] == 0.5
