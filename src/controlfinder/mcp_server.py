"""An MCP server, so AI assistants can find the NIST SP 800-53 controls that answer a security question.

Three read-only tools:
  - search_controls: the controls that fit a question (hybrid, vector, or keyword search)
  - get_control: a control's statement, guidance, and assessment objectives
  - answer_question: a cited answer from the local model, or a fixed sentence when the catalog doesn't cover it

It talks over stdio, so the assistant on the same computer starts it:

    python -m controlfinder.mcp_server

It needs Ollama with nomic-embed-text, and qwen3:8b for answer_question. It searches in memory, which step 5
showed gives the same rankings as the database, so it needs no database. Nothing leaves this computer.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from controlfinder import answer
from controlfinder.catalog import load_catalog
from controlfinder.search import InMemorySearcher, Unavailable

MAX_QUESTION = 2000  # characters, as in the API
MAX_K = 20
ID = re.compile(r"^([a-z]{2})-(\d{1,2})(?:\((\d{1,2})\)|\.(\d{1,2}))?$", re.I)

INSTRUCTIONS = """Find the NIST SP 800-53 Rev 5.2.0 security controls that answer a question, and read them.
- Cite controls by their label (for example IA-5(1)), and describe only what a control's own text says; get_control returns it.
- answer_question answers only from the controls it was shown. When it says a question isn't covered, tell the person so, and don't present general knowledge as NIST's.
- Search results are candidates, not a compliance decision: which controls a system needs depends on its baseline and risk assessment.
- Not affiliated with or endorsed by NIST."""
SEARCH_NOTE = "Candidates ranked by how well they fit the question, not a compliance decision. Read a control's text with get_control before relying on it."
ANSWER_NOTE = (f"From a local model ({answer.CHOSEN['model']}) that saw only the shown controls' text; its citations were measured against "
               "NIST's own mappings (docs/results-answers.md). Check the cited controls' text before relying on it.")

server = MCPServer("control-finder", title="Control Finder", instructions=INSTRUCTIONS, version="0.1.0")
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

Question = Annotated[str, Field(min_length=1, max_length=MAX_QUESTION, description="The question, in plain English.")]
ControlId = Annotated[str, Field(min_length=3, max_length=20, description='A control\'s id or label, such as "ia-5.1" or "IA-5(1)".')]
Method = Literal["hybrid", "vector", "keyword"]
HowMany = Annotated[int, Field(ge=1, le=MAX_K, description="How many controls to return.")]


class Found(TypedDict):
    rank: int
    id: str
    label: str
    title: str
    family: str
    keyword_rank: int | None  # its rank in the keyword top 100 (keyword and hybrid)
    vector_rank: int | None  # its rank in the vector top 100 (vector and hybrid)


class SearchResults(TypedDict):
    question: str
    method: str
    results: list[Found]
    note: str


class ControlText(TypedDict):
    id: str
    label: str
    title: str
    family: str
    parent: str | None  # the base control, for an enhancement
    statement: str
    guidance: str
    assessment_objectives: list[str]  # what an assessor checks (SP 800-53A Rev 5.2.0)


class Cited(TypedDict):
    id: str
    label: str
    title: str


class Answer(TypedDict):
    question: str
    covered: bool
    answer: str  # the model's answer, or a fixed sentence when none of the shown controls answer the question
    citations: list[Cited]
    shown: list[str]  # the labels of the controls the model was shown
    note: str


# How the tools search and answer. Tests replace both.
searcher: InMemorySearcher | None = None
answerer = answer.ask_chosen


def current() -> InMemorySearcher:
    global searcher
    if searcher is None:
        searcher = InMemorySearcher()
    return searcher


def to_id(text: str) -> str | None:
    """'IA-5(1)', 'ia-5.1', and 'IA-05(01)' all become 'ia-5.1'."""
    m = ID.match(text.strip())
    if not m:
        return None
    family, number, enhancement = m.group(1).lower(), int(m.group(2)), m.group(3) or m.group(4)
    return f"{family}-{number}" + (f".{int(enhancement)}" if enhancement else "")


def nonempty(question: str) -> str:
    if not question.strip():
        raise ToolError("The question is empty.")
    return question.strip()


@server.tool(title="Search the controls", annotations=READ_ONLY)
def search_controls(question: Question, method: Method = "hybrid", k: HowMany = 10) -> SearchResults:
    """Find the SP 800-53 controls and enhancements that best fit a question.

    hybrid (the default) fuses keyword and meaning-based search, the best measured on NIST's CSF 2.0
    mappings. keyword works even without Ollama. Each result gives its rank in the keyword and vector lists.
    """
    q, s = nonempty(question), current()
    try:
        results, _ = s.search(q, method, k)
    except Unavailable as exc:
        raise ToolError(f"{exc}; keyword search still works (method=keyword).") from exc
    found = [{"rank": r["rank"], "id": r["id"], "label": s.by_id[r["id"]].label, "title": s.by_id[r["id"]].title,
              "family": s.by_id[r["id"]].family, "keyword_rank": r["keyword_rank"], "vector_rank": r["vector_rank"]} for r in results]
    return {"question": q, "method": method, "results": found, "note": SEARCH_NOTE}


@server.tool(title="Read a control", annotations=READ_ONLY)
def get_control(control_id: ControlId) -> ControlText:
    """A control's statement, guidance, and SP 800-53A assessment objectives (what an assessor checks)."""
    cid = to_id(control_id)
    c = current().by_id.get(cid) if cid else None
    if c is None:
        gone = next((w for w in load_catalog() if w.id == cid and w.withdrawn), None) if cid else None
        raise ToolError(f"{gone.label} is withdrawn in SP 800-53 Rev 5.2.0." if gone else
                        f"No control or enhancement has the id {control_id!r}.")
    return {"id": c.id, "label": c.label, "title": c.title, "family": c.family, "parent": c.parent,
            "statement": c.statement, "guidance": c.guidance, "assessment_objectives": list(c.objectives)}


@server.tool(title="Answer with citations", annotations=READ_ONLY)
def answer_question(question: Question) -> Answer:
    """Answer a security question from the controls the search finds, citing them, with a local model.

    The answer uses only the shown controls' text, and its citations can only name them. When none of them
    answer the question, it says so with a fixed sentence instead of the model's own words.
    """
    q, s = nonempty(question), current()
    try:
        results, _ = s.search(q, "hybrid", answer.CHOSEN["shown"])
    except Unavailable as exc:
        raise ToolError(str(exc)) from exc
    shown = [s.by_id[r["id"]] for r in results]
    try:
        reply = answerer(q, shown)
    except OSError as exc:
        raise ToolError(f"The answer model isn't reachable (Ollama with {answer.CHOSEN['model']}).") from exc
    parsed = answer.parse(reply["raw"], shown)
    if parsed is None:
        raise ToolError("The model's answer wasn't valid JSON. Try again.")
    covered, text, cited = parsed
    return {"question": q, "covered": covered, "answer": answer.shown_to_people(covered, text),
            "citations": [{"id": c, "label": s.by_id[c].label, "title": s.by_id[c].title} for c in cited],
            "shown": [c.label for c in shown], "note": ANSWER_NOTE}


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
