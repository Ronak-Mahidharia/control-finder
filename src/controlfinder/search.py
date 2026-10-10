# ============================================================================
# The search behind the API (step 5): keyword (BM25, in memory), vector
# (exact search in PostgreSQL with pgvector), and hybrid (reciprocal rank
# fusion of the keyword and vector top 100, the vector list weighted 3 times,
# as chosen on CSF 2.0 dev questions in step 4).
# A question is embedded with the same local model and query prefix as the
# published results, so the API returns the rankings that were measured.
# ============================================================================
import logging
import queue
import threading
import time
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from pg8000.exceptions import DatabaseError, InterfaceError

from controlfinder import db
from controlfinder.bm25 import BM25
from controlfinder.catalog import active_controls
from controlfinder.embeddings import QUERY, committed_controls, embed
from controlfinder.fusion import rrf
from controlfinder.vector import search as nearest_in_memory

DEPTH = 100  # each list's length before fusion, as in steps 3 and 4
RRF_K = 60
VECTOR_WEIGHT = 3.0  # chosen on CSF 2.0 dev questions in step 4 (results/rerank.json)
METHODS = ("hybrid", "vector", "keyword")
NO_DATABASE = "the database isn't reachable (docker compose up -d --wait)"
log = logging.getLogger(__name__)


class Unavailable(Exception):
    """The embedding model or the database can't be reached."""


@lru_cache(maxsize=1024)
def _embedded(question):
    vectors, _ = embed([question], QUERY)
    return tuple(vectors[0].tolist())


def embed_question(question):
    """The question as a unit vector from the local model. A repeated question comes from a cache."""
    try:
        return np.asarray(_embedded(question), dtype=np.float32)
    except OSError as exc:  # Ollama isn't running, or the model isn't pulled
        raise Unavailable("the embedding model isn't reachable (Ollama with nomic-embed-text)") from exc


class Pool:
    """A few database connections shared by the API's worker threads, because a pg8000 connection isn't
    thread-safe. Each one connects on first use, so the API starts even while the database is down.
    Last in, first out: one request at a time keeps reusing one open connection."""

    def __init__(self, size=4, url=None):
        self._url = url
        self._free = queue.LifoQueue()
        for _ in range(size):
            self._free.put(None)

    @contextmanager
    def connection(self):
        conn = self._free.get()
        try:
            if conn is None:
                conn = db.connect(self._url)
            yield conn
        except Exception as exc:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = None  # a fresh connection next time
            if isinstance(exc, (OSError, InterfaceError)):
                self._drop_idle()  # the connection was lost, so the database restarted: the idle ones are stale too
            raise
        finally:
            self._free.put(conn)

    def _drop_idle(self):
        idle = []
        while True:
            try:
                idle.append(self._free.get_nowait())
            except queue.Empty:
                break
        for conn in idle:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._free.put(None)

    def close(self):
        while not self._free.empty():
            conn = self._free.get_nowait()
            if conn is not None:
                conn.close()


def _ms(start):
    return (time.perf_counter() - start) * 1000


class Searcher:
    def __init__(self, pool, embed=embed_question):
        self.controls = active_controls()
        self.by_id = {c.id: c for c in self.controls}
        self.bm25 = BM25([f"{c.title}. {c.statement} {c.guidance}" for c in self.controls])
        self.pool, self.embed = pool, embed
        self.database = None  # "ready" or "loaded" once the controls are known to be in the database
        self._preparing = threading.Lock()

    def _run(self, work):
        """work(conn) on a pooled connection. A lost connection is tried once more on a fresh one, because
        the pool's connections go stale when the database restarts."""
        try:
            with self.pool.connection() as conn:
                return work(conn)
        except (OSError, InterfaceError):
            with self.pool.connection() as conn:
                return work(conn)

    def prepare_database(self):
        """Loads the controls into the database unless they're there already: "ready", "loaded", or "unreachable".
        Once it has worked it isn't done again, and a database that starts after the API is prepared on first use."""
        with self._preparing:
            if self.database is None:
                try:
                    loaded = self._run(lambda conn: db.ensure_loaded(conn, self.controls, committed_controls(self.controls)))
                except (OSError, InterfaceError):
                    return "unreachable"
                except DatabaseError as exc:  # it answered but refused: still starting up, or a wrong password
                    log.warning("the database refused: %s", exc)
                    return "unreachable"
                self.database = "loaded" if loaded else "ready"
            return self.database

    def database_versions(self):
        try:
            return self._run(db.versions)
        except (OSError, InterfaceError, DatabaseError):
            return None

    def database_status(self):
        """The database as it is now, for /health: ("unreachable", {}), or ("ready" or "loaded", its versions)."""
        versions = self.database_versions()
        if not versions:
            return "unreachable", {}
        return self.prepare_database(), versions

    def keyword(self, question):
        return [self.controls[i].id for i, _ in self.bm25.search(question, DEPTH)]

    def vector(self, vector):
        if self.database is None and self.prepare_database() == "unreachable":
            raise Unavailable(NO_DATABASE)
        try:
            return self._run(lambda conn: db.nearest(conn, vector, DEPTH, exact=True))
        except (OSError, InterfaceError) as exc:
            raise Unavailable(NO_DATABASE) from exc
        except DatabaseError as exc:  # for example, its table was dropped: the next search checks and fills it again
            self.database = None
            log.warning("vector search failed in the database: %s", exc)
            raise Unavailable("the database isn't ready; try again in a moment") from exc

    def search(self, question, method="hybrid", k=10):
        """The top k, each with its rank in each list that was searched, and the milliseconds each part took.
        Keyword search can return fewer than k when few controls share the question's words."""
        lists, timings = {}, {}
        if method in ("keyword", "hybrid"):
            start = time.perf_counter()
            lists["keyword"] = self.keyword(question)
            timings["keyword"] = _ms(start)
        if method in ("vector", "hybrid"):
            start = time.perf_counter()
            vector = self.embed(question)
            timings["embed"] = _ms(start)
            start = time.perf_counter()
            lists["vector"] = self.vector(vector)
            timings["vector"] = _ms(start)
        if method == "hybrid":
            start = time.perf_counter()
            order = rrf([lists["keyword"], lists["vector"]], k=RRF_K, depth=DEPTH, weights=[1.0, VECTOR_WEIGHT])
            timings["fuse"] = _ms(start)
        else:
            order = lists[method]
        ranks = {name: {cid: i + 1 for i, cid in enumerate(ids)} for name, ids in lists.items()}
        results = [{"rank": i + 1, "id": cid, "keyword_rank": ranks.get("keyword", {}).get(cid),
                    "vector_rank": ranks.get("vector", {}).get(cid)} for i, cid in enumerate(order[:k])]
        return results, timings


class InMemorySearcher(Searcher):
    """The same search, with the vectors searched in memory from the committed embeddings instead of in the
    database. Step 5 showed both give the same top 10 for every question, and memory is faster at this size,
    so the MCP server needs no database. (The API uses the database.)"""

    def __init__(self, embed=embed_question):
        super().__init__(pool=None, embed=embed)
        self.vectors = committed_controls(self.controls)

    def vector(self, vector):
        row = nearest_in_memory(np.asarray(vector, dtype=np.float32)[None, :], self.vectors, DEPTH)[0]
        return [self.controls[i].id for i in row]
