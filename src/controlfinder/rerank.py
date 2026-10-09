# ============================================================================
# Reranking with a cross-encoder: a model reads the question and one control
# together and scores how well they match. That's more careful than search,
# but slower, so it only reorders the top candidates from the first search.
#
# Each model runs on this computer's CPU (the same numbers every run), at a
# pinned version from Hugging Face. All three have permissive licenses
# (checked Oct 8, 2026) and ship as safetensors files.
# Optional: needs requirements-rerank.txt (PyTorch). CI checks the published
# scores without it.
# ============================================================================
MODELS = {
    # name: (pinned revision, license)
    "cross-encoder/ms-marco-MiniLM-L6-v2": ("233902d25c440f23af6f7d6e94d2946bac0bee0a", "Apache-2.0"),
    "mixedbread-ai/mxbai-rerank-xsmall-v1": ("b5c6e9da73abc3711f593f705371cdbe9e0fe422", "Apache-2.0"),
    "BAAI/bge-reranker-base": ("2cfc18c9415c912f9d8155881c133215df768a70", "MIT"),
}
MAX_TOKENS = 512  # question and control together; longer pairs are cut


def load(name):
    from sentence_transformers import CrossEncoder  # imported here, so the rest of the package doesn't need PyTorch

    revision, _ = MODELS[name]
    return CrossEncoder(name, revision=revision, device="cpu", max_length=MAX_TOKENS)


def score_pairs(model, pairs, batch_size=32):
    """One relevance score per (question, control text) pair; higher is better.

    Raw scores (logits), not the default sigmoid: near-certain matches would all round to 1.0
    and tie. The order is the same either way.
    """
    import torch

    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False, activation_fn=torch.nn.Identity())
    return [round(float(s), 6) for s in scores]


def reorder(candidates, scores):
    """Candidates sorted by reranker score; ties keep the first search's order."""
    return [c for _, _, c in sorted(zip((-s for s in scores), range(len(candidates)), candidates))]
