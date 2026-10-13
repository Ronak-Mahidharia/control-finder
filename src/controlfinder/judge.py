# ============================================================================
# An AI judge, checked against NIST's answer keys. A local model reads a
# question, the controls the search found for it, and the cited answer people
# were shown (step 6), and says whether the answer is right: the controls it
# cites address the question, or, when it says the question isn't covered,
# none of the controls found do.
# The answer keys decide the same thing another way (truth()): an answer is
# right when it cites a control NIST links to the question (an anchor, for a
# plain-English question), or says "not covered" to a question SP 800-53
# doesn't answer. So the judge's verdicts can be scored against the keys, to
# see how far a judge could stand in for a key where there is none.
# The judge's reason comes before its verdict in the JSON, so it writes it
# before it decides. One prompt, written before any judge ran; temperature 0
# and a fixed seed, as in step 6.
# ============================================================================
import json
import time

from controlfinder import answer

SYSTEM = ("You check answers to security questions. Judge only from the NIST SP 800-53 Rev 5.2.0 controls you are "
          "given, not from what you know.")
ASK = ("Is the answer right? It is right when the controls it cites address the question. When it says the question "
       "isn't covered, it is right only when none of the controls found address the question.")
FORMAT = 'Answer in JSON: {"reason": "one or two sentences", "right": true or false}'
SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "right": {"type": "boolean"}},
          "required": ["reason", "right"]}
NIST = ("csf2", "sp800_171")


def prompt(question, shown, covered, text, cited):
    controls = "\n\n".join(f"[{c.label}]{' (cited)' if c.id in cited else ''} {c.title}: "
                           f"{c.statement[:answer.STATEMENT_CHARS]}" for c in shown)
    given = (f"{text}\nCites: {', '.join(c.label for c in shown if c.id in cited) or 'nothing'}" if covered
             else answer.NOT_COVERED)
    return (f"Question: {question}\n\nNIST SP 800-53 controls the search found for it:\n\n{controls}\n\n"
            f"The answer people were shown:\n{given}\n\n{ASK} {FORMAT}")


def ask(question, shown, covered, text, cited, model):
    """The judge's raw JSON verdict, with its time and token counts."""
    payload = {"model": model, "stream": False, "format": SCHEMA, "options": answer.OPTIONS,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt(question, shown, covered, text, cited)}]}
    if answer.thinks(model):
        payload["think"] = False
    started = time.perf_counter()
    response = answer._post(answer.CHAT_URL, payload)
    return {"raw": response.get("message", {}).get("content", ""), "seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": response.get("prompt_eval_count", 0), "output_tokens": response.get("eval_count", 0)}


def verdict(raw):
    """True (right), False (wrong), or None when the reply isn't the JSON the schema asks for."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data["right"] if isinstance(data, dict) and isinstance(data.get("right"), bool) else None


def truth(key, question, covered, cited):
    """Whether the answer is right by the answer keys (as step 6 scores it)."""
    if key in NIST:
        return bool(set(cited) & question.expected)
    if key == "plain_english":
        return covered and bool(set(cited) & question.expected)
    return not covered


def agreement(pairs):
    """pairs: (the key says right, the judge says right, or None for an invalid reply). How often they agree, Cohen's
    kappa (agreement beyond chance), and how the judge does at flagging the answers the key calls wrong."""
    valid = [(k, j) for k, j in pairs if j is not None]
    n = len(valid)
    agree = sum(k == j for k, j in valid)
    key_right, judge_right = sum(k for k, _ in valid), sum(j for _, j in valid)
    chance = (key_right * judge_right + (n - key_right) * (n - judge_right)) / (n * n) if n else 0.0
    observed = agree / n if n else 0.0
    flagged = sum(not j for _, j in valid)
    caught = sum(not k and not j for k, j in valid)
    wrong = n - key_right
    return {"answers": len(pairs), "invalid": len(pairs) - n, "key_says_wrong": wrong, "judge_says_wrong": flagged,
            "agreement": round(observed, 3),
            "kappa": round((observed - chance) / (1 - chance), 3) if chance < 1 else 0.0,
            "wrong_answers_caught": round(caught / wrong, 3) if wrong else None,
            "flags_that_are_wrong_answers": round(caught / flagged, 3) if flagged else None,
            "right_answers_flagged": sum(k and not j for k, j in valid)}
