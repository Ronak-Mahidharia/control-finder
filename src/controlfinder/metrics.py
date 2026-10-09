# ============================================================================
# Retrieval metrics. Each function takes one question's ranked control ids
# (best first) and the set of ids NIST says are right.
#   recall@k     share of the right controls found in the top k
#   precision@k  share of the top k that are right
#   hit@k        1 if any right control is in the top k, else 0
#   MRR          1 / the rank of the first right control (0 if none), averaged
# ============================================================================


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def hit_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked, relevant):
    for rank, control_id in enumerate(ranked, start=1):
        if control_id in relevant:
            return 1.0 / rank
    return 0.0


def summarize(results, ks=(1, 5, 10)):
    """Mean of each metric over (ranked, relevant) pairs, rounded to 3 places."""
    n = len(results)
    out = {"questions": n, "mrr": round(sum(reciprocal_rank(r, rel) for r, rel in results) / n, 3)}
    for k in ks:
        out[f"recall@{k}"] = round(sum(recall_at_k(r, rel, k) for r, rel in results) / n, 3)
        out[f"hit@{k}"] = round(sum(hit_at_k(r, rel, k) for r, rel in results) / n, 3)
    out["precision@5"] = round(sum(precision_at_k(r, rel, 5) for r, rel in results) / n, 3)
    return out
