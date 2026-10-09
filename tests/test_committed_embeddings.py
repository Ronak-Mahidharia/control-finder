# The committed embeddings (embeddings/) must be the ones behind the published vector results,
# so step 5 and CI can use them without the embedding model.
import hashlib
import json
from pathlib import Path

import numpy as np

from controlfinder.answer_key import load_csf_questions, load_out_of_scope, load_plain_questions, load_questions
from controlfinder.catalog import active_controls
from controlfinder.embeddings import QUERY
from controlfinder.vector import rank_gap, search

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "embeddings"
MANIFEST = json.loads((FOLDER / "manifest.json").read_text())
COMMITTED = json.loads((ROOT / "results" / "vector_top100.json").read_text())[MANIFEST["controls"]["text"]]
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions}


def vectors(entry):
    v = np.load(FOLDER / entry["file"])
    assert v.dtype == np.float32 and v.shape == (len(entry["ids"]), MANIFEST["dimensions"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    return v


def test_files_match_their_hashes():
    assert set(MANIFEST["sha256"]) == {p.name for p in FOLDER.glob("*.npy")}
    for name, digest in MANIFEST["sha256"].items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == digest, name


def test_ids_follow_the_catalog_and_both_answer_keys():
    assert MANIFEST["controls"]["ids"] == [c.id for c in active_controls()]
    for key, load in KEYS.items():
        assert MANIFEST["questions"][key]["ids"] == [q.id for q in load()]


def test_searching_them_gives_the_published_lists():
    # Float32 rounding differs between computers, so neighbors whose similarities are within 1e-6
    # may swap. Every rank's similarity must match, and the top 10 (where neighbors are at least
    # 3.5e-6 apart) must match exactly.
    docs = vectors(MANIFEST["controls"])
    ids = MANIFEST["controls"]["ids"]
    for key in KEYS:
        entry = MANIFEST["questions"][key]
        questions = vectors(entry)
        for qid, v, row in zip(entry["ids"], questions, search(questions, docs, 100)):
            got, want = [ids[i] for i in row], COMMITTED[key][qid]
            assert got[:10] == want[:10], qid
            similarity = dict(zip(ids, docs.astype(np.float64) @ v.astype(np.float64)))
            assert rank_gap(got, want, similarity) < 1e-6, qid


def test_rank_gap_measures_swaps_only():
    similarity = {"a": 0.9, "b": 0.8, "c": 0.8000001}
    assert rank_gap(["a", "b", "c"], ["a", "b", "c"], similarity) == 0.0
    assert rank_gap(["a", "c", "b"], ["a", "b", "c"], similarity) < 1e-6  # a near-tie swapped
    assert rank_gap(["b", "a", "c"], ["a", "b", "c"], similarity) > 0.09  # a real difference


def test_the_hand_made_questions_are_embedded_in_order():
    for key, load in (("plain_english", load_plain_questions), ("out_of_scope", load_out_of_scope)):
        entry = MANIFEST["questions"][key]
        assert entry["ids"] == [q.id for q in load()] and entry["prefix"] == QUERY, key
        vectors(entry)  # the right shape and type, and unit length
