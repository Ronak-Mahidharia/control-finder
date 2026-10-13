# The AI judge and the question rewriting without a model: the prompts, the requests sent to Ollama, reading the
# replies, the answer keys' verdict, the agreement scores, and how rewritten queries are fused.
import json

import pytest

from controlfinder import answer, judge, rewrite
from controlfinder.answer_key import OutOfScope, Question
from controlfinder.catalog import active_controls
from controlfinder.fusion import rrf

BY_ID = {c.id: c for c in active_controls()}
SHOWN = [BY_ID[i] for i in ("ia-5.1", "ia-5.18", "au-9.7")]
PASSWORDS = Question("03.05.07", "Password Management", "How should passwords be stored?", frozenset({"ia-5.1"}), "test")


def fake_ollama(monkeypatch, content, capabilities=("completion", "thinking")):
    sent = []

    def fake_post(url, payload, timeout=600):
        sent.append((url, payload))
        if url == answer.SHOW_URL:
            return {"capabilities": list(capabilities)}
        return {"message": {"content": content}, "prompt_eval_count": 700, "eval_count": 30}
    monkeypatch.setattr(answer, "_post", fake_post)
    answer.thinks.cache_clear()
    return sent


@pytest.fixture(autouse=True)
def fresh_thinks():
    yield
    answer.thinks.cache_clear()


def test_the_judge_sees_the_controls_found_with_the_cited_ones_marked():
    text = judge.prompt("How should passwords be stored?", SHOWN, True, "Store them salted and hashed.", ["ia-5.1"])
    assert "[IA-5(1)] (cited) Password-based Authentication:" in text and "[IA-5(18)] Password Managers:" in text
    assert "Store them salted and hashed.\nCites: IA-5(1)" in text
    assert text.endswith(f"{judge.ASK} {judge.FORMAT}")
    refused = judge.prompt("What does a SOC 2 audit cost?", SHOWN, False, "It costs about $20,000.", [])
    assert answer.NOT_COVERED in refused and "$20,000" not in refused  # people only ever see the fixed sentence


def test_the_judges_request_and_its_verdict(monkeypatch):
    sent = fake_ollama(monkeypatch, '{"reason": "IA-5(1) covers stored passwords.", "right": true}')
    out = judge.ask("How should passwords be stored?", SHOWN, True, "Hash them.", ["ia-5.1"], "qwen3:8b")
    url, payload = sent[-1]
    assert url == answer.CHAT_URL and payload["format"] == judge.SCHEMA and payload["think"] is False
    assert payload["options"] == answer.OPTIONS and payload["messages"][0]["content"] == judge.SYSTEM
    assert list(judge.SCHEMA["properties"]) == ["reason", "right"]  # the reason comes first
    assert judge.verdict(out["raw"]) is True and out["output_tokens"] == 30
    assert judge.verdict('{"reason": "", "right": false}') is False
    assert judge.verdict("not json") is None and judge.verdict('{"right": "yes"}') is None and judge.verdict("[]") is None


def test_the_answer_keys_verdict():
    assert judge.truth("sp800_171", PASSWORDS, True, ["ia-5.18", "ia-5.1"])  # one NIST link is enough
    assert not judge.truth("sp800_171", PASSWORDS, True, ["ia-5.18"])
    assert not judge.truth("csf2", PASSWORDS, False, [])
    assert judge.truth("plain_english", PASSWORDS, True, ["ia-5.1"])
    assert not judge.truth("plain_english", PASSWORDS, False, ["ia-5.1"])
    soc2 = OutOfScope("oos-01", "What does a SOC 2 audit cost?", "security-adjacent", "test")
    assert judge.truth("out_of_scope", soc2, False, []) and not judge.truth("out_of_scope", soc2, True, ["ia-5.1"])


def test_agreement_with_the_key():
    # The key and the judge on 10 answers: they agree on 8; 4 wrong answers, the judge flags 3 of them and 1 right one.
    pairs = ([(True, True)] * 5 + [(True, False)] + [(False, False)] * 3 + [(False, True)] + [(True, None)])
    a = judge.agreement(pairs)
    assert (a["answers"], a["invalid"], a["key_says_wrong"], a["judge_says_wrong"]) == (11, 1, 4, 4)
    assert a["agreement"] == 0.8 and a["wrong_answers_caught"] == 0.75 and a["flags_that_are_wrong_answers"] == 0.75
    assert a["right_answers_flagged"] == 1
    assert a["kappa"] == pytest.approx((0.8 - 0.52) / 0.48, abs=1e-3)  # chance: 0.6 * 0.6 + 0.4 * 0.4
    assert judge.agreement([(True, True)] * 3)["kappa"] == 0.0  # no wrong answers: nothing beyond chance to measure


def test_rewritten_queries_are_read_without_blanks_or_repeats(monkeypatch):
    assert rewrite.queries('{"queries": ["account  management", "", "Account Management", "audit review"]}') == [
        "account management", "audit review"]
    assert rewrite.queries("not json") == [] and rewrite.queries('["a"]') == []
    sent = fake_ollama(monkeypatch, '{"queries": ["a b", "c d", "e f"]}', capabilities=("completion",))
    rewrite.ask("How should passwords be stored?", "granite4:3b")
    rewrite.ask("How should passwords be stored?", "granite4:3b", titles=["Password-based Authentication"])
    first, again = sent[-2][1], sent[-1][1]
    assert first["format"] == rewrite.schema(3) and again["format"] == rewrite.schema(1) and "think" not in first
    assert rewrite.schema(3)["properties"]["queries"]["minItems"] == 3
    assert "- Password-based Authentication" in again["messages"][1]["content"]
    assert "Password-based Authentication" not in first["messages"][1]["content"]


def test_rewritten_queries_are_fused_with_the_questions_own_lists():
    question = (["a", "b", "c"], ["b", "c", "d"])
    extra = [(["e", "a"], ["e", "f"])]
    assert rewrite.fuse(question, [], 0.5, 3.0) == rrf(list(question), weights=[1.0, 3.0])  # step 4's search
    assert rewrite.fuse(question, extra, 0.5, 3.0) == rrf([*question, *extra[0]], weights=[1.0, 3.0, 0.5, 1.5])
    assert rewrite.fuse(question, extra, 1.0, 3.0)[0] == "e"  # a strong rewrite can lead
    assert json.dumps(rewrite.fuse(question, extra, 0.5, 3.0))  # plain ids
