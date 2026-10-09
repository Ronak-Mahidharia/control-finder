# ============================================================================
# Hybrid search with reciprocal rank fusion (RRF; Cormack, Clarke, and
# Büttcher, SIGIR 2009). Each list adds 1 / (k + rank) to every control it
# returns, so a control near the top of either list rises, and one near the top
# of both rises most. k = 60 is the paper's value; it isn't tuned here.
# Only ranks are used, so keyword scores and cosine similarities never need to
# be put on the same scale. Optional weights scale each list's share; with no
# weights, every list counts the same.
# ============================================================================


def rrf(rankings, k=60, depth=100, weights=None):
    """Combine ranked lists of ids into one, best first."""
    weights = weights or [1.0] * len(rankings)
    scores, best_rank = {}, {}
    for ranking, weight in zip(rankings, weights):
        for rank, control_id in enumerate(ranking[:depth], start=1):
            scores[control_id] = scores.get(control_id, 0.0) + weight / (k + rank)
            best_rank[control_id] = min(best_rank.get(control_id, rank), rank)
    # Ties go to the control ranked higher in any one list, then to its id, so the order never changes.
    return sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))
