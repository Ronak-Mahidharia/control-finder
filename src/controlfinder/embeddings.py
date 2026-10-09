# ============================================================================
# Text embeddings from a local model through Ollama: free, and nothing leaves
# this computer.
#
# The model is nomic-embed-text (137M parameters, Apache-2.0, 768 dimensions).
# It was trained with task prefixes, so a control starts with
# "search_document: " and a question with "search_query: ". It reads at most
# 2,048 tokens of a text; Ollama silently cuts the rest, so embed() counts how
# many texts reached that limit.
# Vectors are scaled to length 1, so a dot product is the cosine similarity.
# Results are cached in data/processed/ (never committed), keyed by the exact
# texts and prefix, so any changed text is embedded again.
# ============================================================================
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
MAX_TOKENS = 2048
DOCUMENT = "search_document: "
QUERY = "search_query: "
CACHE = Path(__file__).resolve().parents[2] / "data" / "processed"


def _post(texts):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def embed(texts, prefix):
    """Unit-length vectors, one request per text, so each text's token count is known."""
    vectors, cut = [], 0
    for text in texts:
        out = _post([prefix + text])
        if out.get("prompt_eval_count", 0) >= MAX_TOKENS:
            cut += 1
        vectors.append(out["embeddings"][0])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, cut


def cached_embed(name, texts, prefix):
    """embed(), but read from data/processed/ when these exact texts were embedded before."""
    key = hashlib.sha256(json.dumps([MODEL, prefix, list(texts)]).encode()).hexdigest()[:16]
    path = CACHE / f"{name}-{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["vectors"], int(data["cut"])
    vectors, cut = embed(texts, prefix)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, cut=cut)
    return vectors, cut
