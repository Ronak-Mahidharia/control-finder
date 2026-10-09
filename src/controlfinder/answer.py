# ============================================================================
# Cited answers (step 6): a local model reads the controls the search found,
# answers from them only, and cites each control it relies on, or says the
# question isn't covered.
#
# The model answers in JSON that Ollama holds to a schema: "covered",
# "answer", and "citations". The schema allows only the shown controls'
# labels as citations, so a citation can't name a control that wasn't shown.
# The free-text answer can still mention one; mentioned() finds those.
# When the model says a question isn't covered, its own text often answers
# the question anyway from what the model already knew (measured in step 6),
# so people only ever see NOT_COVERED for those answers.
# Temperature 0 and a fixed seed make answers repeatable, and models that
# "think" before answering (qwen3) have that step turned off, as in Car
# Safety Checker.
# ============================================================================
import json
import re
import time
import urllib.request
from functools import lru_cache

from controlfinder.embeddings import URL as EMBED_URL

CHAT_URL = EMBED_URL.replace("/api/embed", "/api/chat")
SHOW_URL = EMBED_URL.replace("/api/embed", "/api/show")
STATEMENT_CHARS = 700  # each control's statement is cut here, to keep the prompt short
SYSTEM = "You answer security questions using only the NIST SP 800-53 Rev 5.2.0 controls you are given."
# Prompt wordings, compared on dev questions only.
PROMPTS = {
    "v1": ("Which of these controls address the question? Cite only controls from the list. If none of them address it, "
           "set covered to false and leave citations empty."),
    "v2": ("Cite only the controls whose text directly answers the question, most relevant first, and leave out controls "
           "that are only on a related subject. If none of them answer it, set covered to false and leave citations empty."),
}
FORMAT = 'Answer in JSON: {"covered": true or false, "answer": "...", "citations": ["..."]}'
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
LABEL = re.compile(r"\b([A-Z]{2})-(\d{1,2})(?:\((\d{1,2})\))?(?!\d)")
NOT_COVERED = "SP 800-53 Rev 5.2.0 doesn't cover this question: none of the controls found for it answer it."
EXPLAINS = re.compile(r"(?:do|does|did) not address|n't address|not addressed|none of the (?:provided|given|listed)", re.I)
# The setup chosen on dev questions in step 6 (results/answers.json), which the API uses.
CHOSEN = {"model": "qwen3:8b", "version": "v1", "shown": 10}


def prompt(question, shown, version):
    controls = "\n\n".join(f"[{c.label}] {c.title}: {c.statement[:STATEMENT_CHARS]}" for c in shown)
    return (f"NIST SP 800-53 controls found for this question:\n\n{controls}\n\nQuestion: {question}\n\n"
            f"{PROMPTS[version]} {FORMAT}")


def schema(labels):
    return {"type": "object",
            "properties": {"covered": {"type": "boolean"}, "answer": {"type": "string"},
                           "citations": {"type": "array", "items": {"type": "string", "enum": list(labels)}}},
            "required": ["covered", "answer", "citations"]}


def _post(url, payload, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


@lru_cache(maxsize=None)
def thinks(model):
    """Whether the model "thinks" before answering, a step turned off here for speed."""
    return "thinking" in _post(SHOW_URL, {"model": model}, timeout=60).get("capabilities", [])


def ask(question, shown, model, version):
    """The model's raw JSON answer, with its time and token counts."""
    payload = {"model": model, "stream": False, "format": schema(c.label for c in shown),
               "options": OPTIONS,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(question, shown, version)}]}
    if thinks(model):
        payload["think"] = False
    started = time.perf_counter()
    response = _post(CHAT_URL, payload)
    return {"raw": response.get("message", {}).get("content", ""), "seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": response.get("prompt_eval_count", 0), "output_tokens": response.get("eval_count", 0)}


def parse(raw, shown):
    """(covered, answer, cited ids) from the model's JSON, or None if it isn't valid. Citations keep the
    model's order without repeats. An answer that says it isn't covered counts as citing nothing."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    by_label = {c.label: c.id for c in shown}
    cited = []
    for label in data.get("citations") or []:
        control_id = by_label.get(label)
        if control_id and control_id not in cited:
            cited.append(control_id)
    covered = data.get("covered") is True
    return covered, str(data.get("answer", "")), cited if covered else []


def mentioned(text):
    """Control labels written in the answer's text, as NIST prints them ("AC-02" counts as "AC-2")."""
    return {f"{family}-{int(number)}" + (f"({int(enhancement)})" if enhancement else "")
            for family, number, enhancement in LABEL.findall(text)}


def citation_scores(cited, expected):
    """Precision, recall, and F1 of one answer's citations. Citing nothing scores 0 on all three."""
    hits = len(set(cited) & set(expected))
    if not hits:
        return 0.0, 0.0, 0.0
    precision, recall = hits / len(cited), hits / len(expected)
    return precision, recall, 2 * precision * recall / (precision + recall)


def refusal_text(text):
    """How the model's own text reads when it says a question isn't covered: "empty", "explains" (says the
    controls don't address it), or "answers anyway" (answers the question from what the model already knew)."""
    if not text.strip():
        return "empty"
    return "explains" if EXPLAINS.search(text) else "answers anyway"


def shown_to_people(covered, text):
    """The answer text people see: the model's own words only when it found the question covered."""
    return text if covered else NOT_COVERED


def ask_chosen(question, shown):
    """ask(), with the model and prompt chosen in step 6."""
    return ask(question, shown, CHOSEN["model"], CHOSEN["version"])
