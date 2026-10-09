# ============================================================================
# Vector search: rank controls by how close their meaning is to the question,
# using unit-length embeddings, where a dot product is the cosine similarity.
# ============================================================================
import numpy as np


def search(query_vectors, document_vectors, k):
    """For each query, the indexes of the k closest documents, best first. Ties keep catalog order."""
    scores = query_vectors @ document_vectors.T
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order.tolist()


def rank_gap(got, want, similarity):
    """The largest similarity difference between the ids at the same rank of two rankings. Two searches
    that differ only by float32 rounding give about 1e-7: they can swap near-tied neighbors, nothing more."""
    return max((abs(similarity[a] - similarity[b]) for a, b in zip(got, want, strict=True)), default=0.0)
