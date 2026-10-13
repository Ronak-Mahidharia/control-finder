# ============================================================================
# Searching with the question in NIST's words. A local model rewrites the
# question as 3 short search queries in the words SP 800-53 uses ("account
# management", "audit record review"); each query is searched by keyword and
# by meaning, and every list is fused with the question's own (weighted
# reciprocal rank fusion, as in step 4). Search again: the model then reads
# the titles of the top 5 results and writes 1 more query for a control the
# question needs that they miss, which is searched and fused the same way.
# Queries are JSON held to a schema; temperature 0 and a fixed seed.
# ============================================================================
import json
import time

from controlfinder import answer
from controlfinder.fusion import rrf

SYSTEM = "You turn security questions into search queries for the NIST SP 800-53 Rev 5.2.0 catalog of controls."
REWRITE = ("Write 3 short search queries (2 to 6 words each) in the words NIST SP 800-53 uses for its controls, for "
           "the controls this question needs.")
AGAIN = ("These controls were found for it:\n{titles}\n\nWrite 1 more short search query (2 to 6 words) in SP 800-53's "
         "words, for a control the question needs that these miss.")
AGAIN_FROM = 5  # search again reads the titles of this many top results


def schema(n):
    return {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"},
                                                         "minItems": n, "maxItems": n}}, "required": ["queries"]}


def prompt(question, titles=None):
    if titles is None:
        return f"Question: {question}\n\n{REWRITE} " + 'Answer in JSON: {"queries": ["...", "...", "..."]}'
    listed = "\n".join(f"- {t}" for t in titles)
    return f"Question: {question}\n\n{AGAIN.format(titles=listed)} " + 'Answer in JSON: {"queries": ["..."]}'


def ask(question, model, titles=None):
    """The model's raw JSON queries (3, or 1 to search again), with its time and token counts."""
    payload = {"model": model, "stream": False, "format": schema(3 if titles is None else 1), "options": answer.OPTIONS,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(question, titles)}]}
    if answer.thinks(model):
        payload["think"] = False
    started = time.perf_counter()
    response = answer._post(answer.CHAT_URL, payload)
    return {"raw": response.get("message", {}).get("content", ""), "seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": response.get("prompt_eval_count", 0), "output_tokens": response.get("eval_count", 0)}


def queries(raw):
    """The queries in a reply, without blanks or repeats (case aside); none when it isn't valid JSON."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out, seen = [], set()
    for q in (data.get("queries") or []) if isinstance(data, dict) else []:
        q = " ".join(str(q).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def fuse(original, extra, weight, vector_weight, k=60, depth=100):
    """The question's keyword and vector lists, then each extra query's, fused. original and each item of extra are
    (keyword list, vector list); an extra query's lists count `weight` times as much as the question's."""
    rankings, weights = [], []
    for (keyword, vector), w in [(original, 1.0)] + [(lists, weight) for lists in extra]:
        rankings += [keyword, vector]
        weights += [w, w * vector_weight]
    return rrf(rankings, k=k, depth=depth, weights=weights)
