"""The MCP server, tested through a real MCP client connected in-process. The committed question embeddings
stand in for the embedding model and a stand-in replaces the answer model, so CI needs no Ollama."""
import json
from pathlib import Path

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from controlfinder import answer, mcp_server  # noqa: E402
from controlfinder.answer_key import load_csf_questions, load_out_of_scope, load_plain_questions, load_questions  # noqa: E402
from controlfinder.search import InMemorySearcher, Unavailable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "embeddings" / "manifest.json").read_text())
KEYS = {"csf2": load_csf_questions, "sp800_171": load_questions, "plain_english": load_plain_questions,
        "out_of_scope": load_out_of_scope}


@pytest.fixture(scope="module")
def stand_in_searcher():
    embedded = {}
    for key, load in KEYS.items():
        texts = {q.id: q.text for q in load()}
        entry = MANIFEST["questions"][key]
        embedded.update(zip((texts[i] for i in entry["ids"]), np.load(ROOT / "embeddings" / entry["file"])))

    def embed(question):
        if question not in embedded:
            raise Unavailable("no embedding model in this test")
        return embedded[question]
    return InMemorySearcher(embed=embed)


@pytest.fixture(autouse=True)
def replies(monkeypatch, stand_in_searcher):
    """What the stand-in answer model says next."""
    queue = []

    def answerer(question, shown):
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"raw": reply(shown) if callable(reply) else reply}
    monkeypatch.setattr(mcp_server, "searcher", stand_in_searcher)
    monkeypatch.setattr(mcp_server, "answerer", answerer)
    return queue


def call(tool, arguments):
    async def go():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(tool, arguments)
    return anyio.run(go)


def listed_tools():
    async def go():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools
    return {tool.name: tool for tool in anyio.run(go)}


def test_three_read_only_tools_with_input_limits():
    tools = listed_tools()
    assert set(tools) == {"search_controls", "get_control", "answer_question"}
    for t in tools.values():
        assert t.annotations.read_only_hint and t.annotations.destructive_hint is False and t.annotations.open_world_hint is False
        assert t.output_schema
    question = tools["search_controls"].input_schema["properties"]["question"]
    assert (question["minLength"], question["maxLength"]) == (1, mcp_server.MAX_QUESTION)
    assert tools["search_controls"].input_schema["properties"]["k"]["maximum"] == mcp_server.MAX_K


def test_search_gives_the_published_rankings():
    vector = json.loads((ROOT / "results" / "vector_top100.json").read_text())["title + statement + guidance"]
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]

    async def go():
        async with Client(mcp_server.server) as client:
            for key, load in (("csf2", load_csf_questions), ("sp800_171", load_questions)):
                for q in load():
                    hybrid = await client.call_tool("search_controls", {"question": q.text, "k": 20})
                    by_vector = await client.call_tool("search_controls", {"question": q.text, "method": "vector"})
                    assert [r["id"] for r in hybrid.structured_content["results"]] == candidates[key][q.id][:20], q.id
                    assert [r["id"] for r in by_vector.structured_content["results"]] == vector[key][q.id][:10], q.id
    anyio.run(go)


def test_results_carry_labels_and_ranks():
    out = call("search_controls", {"question": "We store customer passwords. How should they be protected?", "k": 3}).structured_content
    assert out["method"] == "hybrid" and len(out["results"]) == 3 and "get_control" in out["note"]
    first = out["results"][0]
    assert set(first) == {"rank", "id", "label", "title", "family", "keyword_rank", "vector_rank"} and first["rank"] == 1


def test_keyword_search_works_without_the_model_and_the_others_say_so():
    keyword = call("search_controls", {"question": "a question with no embedding here", "method": "keyword"})
    assert not keyword.is_error
    hybrid = call("search_controls", {"question": "a question with no embedding here"})
    assert hybrid.is_error and "method=keyword" in hybrid.content[0].text


def test_a_control_by_its_id_or_label():
    for given in ("ia-5.1", "IA-5(1)", "IA-05(01)"):
        out = call("get_control", {"control_id": given}).structured_content
        assert (out["id"], out["label"], out["title"]) == ("ia-5.1", "IA-5(1)", "Password-based Authentication"), given
        assert out["assessment_objectives"] and out["parent"] == "ia-5"
    withdrawn = call("get_control", {"control_id": "AC-2(10)"})
    assert withdrawn.is_error and "AC-2(10) is withdrawn" in withdrawn.content[0].text
    assert call("get_control", {"control_id": "zz-99"}).is_error
    assert call("get_control", {"control_id": "not a label"}).is_error


def test_an_answer_is_shown_the_hybrid_top_10_and_cites_only_those(replies, stand_in_searcher):
    q = load_csf_questions()[0]
    candidates = json.loads((ROOT / "results" / "rerank_scores.json").read_text())["candidates"]["csf2"][q.id]
    replies.append(lambda shown: json.dumps({"covered": True, "answer": "From the controls.", "citations": [shown[0].label]}))
    out = call("answer_question", {"question": q.text}).structured_content
    assert out["shown"] == [stand_in_searcher.by_id[c].label for c in candidates[:answer.CHOSEN["shown"]]]
    assert out["covered"] is True and out["answer"] == "From the controls."
    assert [c["label"] for c in out["citations"]] == [out["shown"][0]]


def test_not_covered_gets_the_fixed_sentence(replies):
    replies.append(json.dumps({"covered": False, "answer": "443", "citations": []}))
    out = call("answer_question", {"question": load_out_of_scope()[5].text}).structured_content
    assert out["covered"] is False and out["answer"] == answer.NOT_COVERED and out["citations"] == []


def test_answer_model_problems_are_named(replies):
    q = load_plain_questions()[1].text
    replies.append(ConnectionRefusedError("Ollama isn't running"))
    down = call("answer_question", {"question": q})
    assert down.is_error and answer.CHOSEN["model"] in down.content[0].text
    replies.append("not json")
    assert call("answer_question", {"question": q}).is_error
    assert call("answer_question", {"question": "   "}).is_error


def test_ids_and_labels_are_read_the_same_way():
    assert mcp_server.to_id("SC-8(1)") == mcp_server.to_id("sc-8.1") == mcp_server.to_id(" SC-08(01) ") == "sc-8.1"
    assert mcp_server.to_id("AC-2") == "ac-2" and mcp_server.to_id("AC-2(") is None and mcp_server.to_id("2") is None
