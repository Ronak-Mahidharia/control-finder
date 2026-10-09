# The answer code without a model: the prompt, the JSON schema, reading the model's JSON, finding
# control labels in its text, scoring citations, and the request sent to Ollama.
import json

import pytest

from controlfinder import answer
from controlfinder.catalog import active_controls

BY_ID = {c.id: c for c in active_controls()}
SHOWN = [BY_ID[i] for i in ("ia-5.1", "ia-5.18", "au-9.7")]


def test_the_prompt_shows_each_control_then_the_question():
    text = answer.prompt("How should passwords be stored?", SHOWN, "v2")
    for c in SHOWN:
        assert f"[{c.label}] {c.title}: {c.statement[:answer.STATEMENT_CHARS]}" in text
    assert text.index("[IA-5(1)]") < text.index("[IA-5(18)]") < text.index("Question: How should passwords be stored?")
    assert text.endswith(f"{answer.PROMPTS['v2']} {answer.FORMAT}")
    long_statement = max(active_controls(), key=lambda c: len(c.statement))
    assert long_statement.statement[:answer.STATEMENT_CHARS + 1] not in answer.prompt("q", [long_statement], "v1")


def test_the_schema_allows_only_the_shown_labels():
    s = answer.schema(c.label for c in SHOWN)
    assert s["properties"]["citations"]["items"]["enum"] == ["IA-5(1)", "IA-5(18)", "AU-9(7)"]
    assert s["required"] == ["covered", "answer", "citations"]


def test_parse_keeps_the_models_order_without_repeats():
    raw = json.dumps({"covered": True, "answer": "Use a salted hash.", "citations": ["IA-5(18)", "IA-5(1)", "IA-5(18)", "ZZ-9"]})
    assert answer.parse(raw, SHOWN) == (True, "Use a salted hash.", ["ia-5.18", "ia-5.1"])


def test_not_covered_cites_nothing_and_bad_json_is_none():
    raw = json.dumps({"covered": False, "answer": "Not covered.", "citations": ["IA-5(1)"]})
    assert answer.parse(raw, SHOWN) == (False, "Not covered.", [])
    assert answer.parse(json.dumps({"covered": "yes", "answer": "", "citations": []}), SHOWN)[0] is False
    assert answer.parse("not json", SHOWN) is None
    assert answer.parse("[1, 2]", SHOWN) is None


def test_mentioned_finds_labels_the_way_nist_prints_them():
    assert answer.mentioned("See AC-02(03), SC-8(1) and IA-5; also PM-11a.") == {"AC-2(3)", "SC-8(1)", "IA-5", "PM-11"}
    assert answer.mentioned("COVID-19 and ac-2 and AC-123") == set()


def test_citation_scores():
    assert answer.citation_scores(["ac-2", "ac-3"], {"ac-2"}) == pytest.approx((0.5, 1.0, 2 / 3))
    assert answer.citation_scores(["ac-2"], {"ac-2", "ac-3", "ac-6", "ac-7"}) == pytest.approx((1.0, 0.25, 0.4))
    assert answer.citation_scores([], {"ac-2"}) == (0.0, 0.0, 0.0)
    assert answer.citation_scores(["si-4"], {"ac-2"}) == (0.0, 0.0, 0.0)


def test_the_request_to_ollama(monkeypatch):
    sent = []

    def fake_post(url, payload, timeout=600):
        sent.append((url, payload))
        if url == answer.SHOW_URL:
            return {"capabilities": ["completion", "thinking"]}
        return {"message": {"content": '{"covered": false, "answer": "", "citations": []}'}, "prompt_eval_count": 900, "eval_count": 12}

    monkeypatch.setattr(answer, "_post", fake_post)
    answer.thinks.cache_clear()
    try:
        out = answer.ask("How should passwords be stored?", SHOWN, "qwen3:8b", "v1")
    finally:
        answer.thinks.cache_clear()
    url, payload = sent[-1]
    assert url == answer.CHAT_URL and payload["model"] == "qwen3:8b" and payload["stream"] is False
    assert payload["think"] is False and payload["options"] == {"temperature": 0, "seed": 42, "num_ctx": 8192}
    assert payload["format"] == answer.schema(c.label for c in SHOWN)
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"] == answer.prompt("How should passwords be stored?", SHOWN, "v1")
    assert (out["prompt_tokens"], out["output_tokens"]) == (900, 12) and out["raw"].startswith('{"covered"')


def test_a_model_that_doesnt_think_gets_no_think_setting(monkeypatch):
    def fake_post(url, payload, timeout=600):
        if url == answer.SHOW_URL:
            return {"capabilities": ["completion", "tools"]}
        fake_post.payload = payload
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(answer, "_post", fake_post)
    answer.thinks.cache_clear()
    try:
        answer.ask("q", SHOWN, "granite4:3b", "v1")
    finally:
        answer.thinks.cache_clear()
    assert "think" not in fake_post.payload


def test_what_a_not_covered_answer_says_is_sorted_three_ways():
    # Examples word for word from the step 6 answers.
    assert answer.refusal_text("") == answer.refusal_text("  ") == "empty"
    assert answer.refusal_text("None of the provided NIST SP 800-53 Rev 5.2.0 controls address the question.") == "explains"
    assert answer.refusal_text("The provided controls do not address the cost of a SOC 2 Type II audit.") == "explains"
    assert answer.refusal_text("The question is not addressed by any of the provided controls.") == "explains"
    assert answer.refusal_text("443") == answer.refusal_text("The capital of Australia is Canberra.") == "answers anyway"


def test_people_never_see_the_models_own_text_for_not_covered():
    assert answer.shown_to_people(False, "443") == answer.NOT_COVERED
    assert answer.shown_to_people(True, "Use a salted hash.") == "Use a salted hash."


def test_the_api_uses_the_setup_chosen_on_dev():
    from pathlib import Path

    chosen = json.loads((Path(__file__).resolve().parent.parent / "results" / "answers.json").read_text())["chosen"]
    assert chosen == f"{answer.CHOSEN['model']}, prompt {answer.CHOSEN['version']}, top {answer.CHOSEN['shown']}"


def test_ask_chosen_uses_the_chosen_model_and_prompt(monkeypatch):
    seen = []
    monkeypatch.setattr(answer, "ask", lambda question, shown, model, version: seen.append((question, model, version)) or {"raw": "{}"})
    answer.ask_chosen("How should passwords be stored?", SHOWN)
    assert seen == [("How should passwords be stored?", answer.CHOSEN["model"], answer.CHOSEN["version"])]
