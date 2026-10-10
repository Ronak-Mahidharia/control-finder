# ============================================================================
# The Control Finder API (step 5): search the 1,014 active SP 800-53 controls
# and enhancements, and read any one of them.
#
# Run:   docker compose up -d --wait            (the database)
#        ollama serve, with nomic-embed-text    (embeds each question)
#        uvicorn controlfinder.api:app --app-dir src --host 127.0.0.1 --port 8000
# Then:  http://127.0.0.1:8000/docs for every endpoint, or for example
#        curl "http://127.0.0.1:8000/search?q=How+should+we+store+passwords%3F&k=5"
#        curl "http://127.0.0.1:8000/answer?q=How+should+we+store+passwords%3F"  (needs qwen3:8b too)
# It also serves the website (web/dist, built with: cd web && npm ci && npm run build)
# at http://127.0.0.1:8000/. At startup it loads the committed embeddings into
# the database if they aren't there. Keyword search works even without the database or Ollama;
# the other methods answer 503 until both can be reached.
# ============================================================================
import json
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from controlfinder import answer
from controlfinder.catalog import load_catalog, to_id
from controlfinder.embeddings import MODEL as EMBED_MODEL
from controlfinder.embeddings import URL as EMBED_URL
from controlfinder.search import DEPTH, VECTOR_WEIGHT, Pool, Searcher, Unavailable

MAX_QUESTION = 2000  # characters; far below the embedding model's 2,048-token limit
SITE = Path(__file__).resolve().parents[2] / "web" / "dist"
MAX_K = 50


class Result(BaseModel):
    rank: int
    id: str = Field(description='OSCAL id, for example "ia-5.1"')
    label: str = Field(description='as NIST prints it, for example "IA-5(1)"')
    title: str
    family: str
    keyword_rank: int | None = Field(description=f"its rank in the keyword top {DEPTH}, if it's there (keyword and hybrid)")
    vector_rank: int | None = Field(description=f"its rank in the vector top {DEPTH}, if it's there (vector and hybrid)")


class SearchResponse(BaseModel):
    query: str
    method: Literal["hybrid", "vector", "keyword"]
    vector_weight: float | None = Field(description="the vector list's weight in the fusion (hybrid only)")
    results: list[Result]


class ControlDetail(BaseModel):
    id: str
    label: str
    title: str
    family: str
    parent: str | None = Field(description="the base control, for an enhancement")
    statement: str
    guidance: str
    assessment_objectives: list[str] = Field(description="what an assessor checks (SP 800-53A Rev 5.2.0)")


class Citation(BaseModel):
    id: str
    label: str
    title: str


class AnswerResponse(BaseModel):
    query: str
    covered: bool
    answer: str = Field(description='the model\'s answer, or a fixed sentence when none of the controls answer the question')
    citations: list[Citation] = Field(description="the controls the answer relies on; only ever ones it was shown")
    shown: list[str] = Field(description="ids of the controls the model was shown, the hybrid search's top results")
    model: str


class Health(BaseModel):
    status: Literal["ok", "degraded"] = Field(description="degraded: only keyword search works right now")
    controls: int
    database: Literal["ready", "loaded", "unreachable"] = Field(
        description="right now: ready (the controls were already there), loaded (this API loaded them), or not reachable")
    postgresql: str | None
    pgvector: str | None
    embedding_model: Literal["reachable", "unreachable"]


def _embedding_model_ready():
    """Ollama is running and has the embedding model: without the model, it can't embed a question."""
    try:
        with urllib.request.urlopen(EMBED_URL.replace("/api/embed", "/api/tags"), timeout=2) as response:
            names = {m.get("name") for m in json.load(response).get("models", [])}
    except (OSError, ValueError):
        return False
    return EMBED_MODEL in names or f"{EMBED_MODEL}:latest" in names


def create_app(searcher=None, embedding_model_check=_embedding_model_ready, answerer=answer.ask_chosen, site=SITE):
    @asynccontextmanager
    async def lifespan(app):
        app.state.searcher = searcher or Searcher(Pool())
        app.state.searcher.prepare_database()  # or on first use, if the database starts later
        yield
        app.state.searcher.pool.close()

    app = FastAPI(title="Control Finder", version="0.1.0", lifespan=lifespan,
                  description="Find the NIST SP 800-53 Rev 5.2.0 controls that answer a security question. "
                              "Not affiliated with or endorsed by NIST; a reference tool, not compliance advice.")

    @app.exception_handler(Unavailable)
    async def unavailable(request: Request, exc: Unavailable):
        return JSONResponse(status_code=503, content={"detail": f"{exc}; keyword search still works (method=keyword)"})

    @app.get("/health", response_model=Health)
    def health(request: Request):
        s = request.app.state.searcher
        database, versions = s.database_status()  # as it is now: it may have stopped, or started, since the API did
        model = "reachable" if embedding_model_check() else "unreachable"
        ok = database != "unreachable" and model == "reachable"
        return {"status": "ok" if ok else "degraded", "controls": len(s.controls), "database": database,
                "postgresql": versions.get("postgresql"), "pgvector": versions.get("pgvector"), "embedding_model": model}

    @app.get("/search", response_model=SearchResponse)
    def search(request: Request, response: Response,
               q: Annotated[str, Query(min_length=1, max_length=MAX_QUESTION, description="the question, in plain English")],
               method: Literal["hybrid", "vector", "keyword"] = "hybrid",
               k: Annotated[int, Query(ge=1, le=MAX_K, description="how many results")] = 10):
        q = q.strip()
        if not q:
            raise HTTPException(status_code=422, detail="q is only spaces")
        s = request.app.state.searcher
        results, timings = s.search(q, method, k)
        response.headers["Server-Timing"] = ", ".join(f"{name};dur={ms:.2f}" for name, ms in timings.items())
        for r in results:
            c = s.by_id[r["id"]]
            r.update(label=c.label, title=c.title, family=c.family)
        return {"query": q, "method": method, "vector_weight": VECTOR_WEIGHT if method == "hybrid" else None, "results": results}

    @app.get("/answer", response_model=AnswerResponse)
    def answer_question(request: Request, response: Response,
                        q: Annotated[str, Query(min_length=1, max_length=MAX_QUESTION, description="the question, in plain English")]):
        q = q.strip()
        if not q:
            raise HTTPException(status_code=422, detail="q is only spaces")
        s = request.app.state.searcher
        results, timings = s.search(q, "hybrid", answer.CHOSEN["shown"])
        shown = [s.by_id[r["id"]] for r in results]
        start = time.perf_counter()
        try:
            reply = answerer(q, shown)
        except OSError as exc:
            raise Unavailable(f"the answer model isn't reachable (Ollama with {answer.CHOSEN['model']})") from exc
        timings["answer"] = (time.perf_counter() - start) * 1000
        parsed = answer.parse(reply["raw"], shown)
        if parsed is None:
            raise HTTPException(status_code=502, detail="the model's answer wasn't valid JSON")
        covered, text, cited = parsed
        response.headers["Server-Timing"] = ", ".join(f"{name};dur={ms:.2f}" for name, ms in timings.items())
        return {"query": q, "covered": covered, "answer": answer.shown_to_people(covered, text),
                "citations": [{"id": c, "label": s.by_id[c].label, "title": s.by_id[c].title} for c in cited],
                "shown": [c.id for c in shown], "model": answer.CHOSEN["model"]}

    @app.get("/controls/{control_id}", response_model=ControlDetail)
    def control(request: Request, control_id: str):
        cid = to_id(control_id)  # "ia-5.1", or NIST's label "IA-5(1)", as the MCP server takes them
        c = request.app.state.searcher.by_id.get(cid) if cid else None
        if c is None:
            gone = next((w for w in load_catalog() if w.id == cid and w.withdrawn), None) if cid else None
            detail = f"{gone.label} is withdrawn in SP 800-53 Rev 5.2.0" if gone else f"no control or enhancement has the id {control_id!r}"
            raise HTTPException(status_code=404, detail=detail)
        return {"id": c.id, "label": c.label, "title": c.title, "family": c.family, "parent": c.parent,
                "statement": c.statement, "guidance": c.guidance, "assessment_objectives": list(c.objectives)}

    # The website, when it's been built. Mounted last, so the API's routes above always come first.
    if site is not None and (Path(site) / "index.html").exists():
        app.mount("/", StaticFiles(directory=site, html=True), name="site")
    return app


app = create_app()
