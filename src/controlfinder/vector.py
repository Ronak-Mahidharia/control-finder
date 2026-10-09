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
