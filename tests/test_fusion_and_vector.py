# Reciprocal rank fusion and vector search, checked against small examples worked out by hand.
import numpy as np

from controlfinder.fusion import rrf
from controlfinder.vector import search


def test_rrf_adds_one_over_k_plus_rank():
    keyword = ["ac-2", "ia-5", "sc-7"]
    vector = ["ia-5", "sc-7", "ac-2"]
    # ia-5: 1/(60+2) + 1/(60+1); sc-7: 1/63 + 1/62; ac-2: 1/61 + 1/63. ia-5 > ac-2 > sc-7.
    assert rrf([keyword, vector]) == ["ia-5", "ac-2", "sc-7"]


def test_rrf_keeps_controls_found_by_only_one_list_and_respects_depth():
    assert rrf([["ac-2"], ["pm-11"]]) == ["ac-2", "pm-11"]  # a tie: same score, same best rank, then by id
    assert "sc-7" not in rrf([["ac-2", "sc-7"], ["ac-2"]], depth=1)


def test_vector_search_orders_by_cosine():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    query = np.array([[0.8, 0.6]], dtype=np.float32)
    # Dot products: 0.8, 0.6, 0.96, so the third document comes first.
    assert search(query, docs, k=3) == [[2, 0, 1]]
    tied = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert search(np.array([[1.0, 0.0]], dtype=np.float32), tied, k=2) == [[0, 1]]  # ties keep catalog order
