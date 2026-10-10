# ============================================================================
# The controls in a vector database: PostgreSQL with the pgvector extension.
#
# One row per active control, with its embedding in a vector(768) column.
# The embeddings are unit length, so the inner product is the cosine
# similarity; pgvector's README advises inner product for vectors like these.
# Two searches:
#   exact: orders by pgvector's inner_product() function, which no index can
#          serve, so every control is compared, like the in-memory search in
#          vector.py. Ties go to catalog order the same way.
#   index: orders by the <#> operator (negative inner product), which the HNSW
#          index can serve. The index visits only part of the table, so it can
#          miss a close control, and PostgreSQL decides whether to use it.
#          use_index() forces it; hnsw.ef_search sets how many candidates it
#          keeps, and that's also the most rows one search returns (default 40).
# Connects to CONTROLFINDER_DB_URL, or else the docker-compose.yml database.
# The database tests and scripts/run_database_check.py rebuild the table, so
# they use a database of their own on the same server (connect_for_checks).
# ============================================================================
import json
import os
import urllib.parse

import pg8000.native
from pg8000.exceptions import DatabaseError

LOCAL_URL = "postgresql://controlfinder:controlfinder-local@127.0.0.1:55432/controlfinder"
CHECK_DATABASE = "controlfinder_check"
DIMENSIONS = 768
HNSW = {"m": 16, "ef_construction": 64}  # pgvector's defaults, written out
INDEX = "controls_embedding_hnsw"
EXACT_SQL = "SELECT id FROM controls ORDER BY inner_product(embedding, CAST(:v AS vector)) DESC, position LIMIT :k"
INDEX_SQL = "SELECT id FROM controls ORDER BY embedding <#> CAST(:v AS vector) LIMIT :k"


def _plain(text):
    return urllib.parse.unquote(text) if text else text  # a URL writes characters like @ in a password as %40


def connect(url=None):
    parts = urllib.parse.urlparse(url or os.environ.get("CONTROLFINDER_DB_URL", LOCAL_URL))
    return pg8000.native.Connection(user=_plain(parts.username), password=_plain(parts.password), host=parts.hostname,
                                    port=parts.port or 5432, database=_plain(parts.path.lstrip("/")), timeout=30)


def connect_for_checks(url=None):
    """A connection to CHECK_DATABASE on the same server, which is made if it's missing. The checks drop and
    rebuild the controls table, so the table the site searches is never touched while they run."""
    url = url or os.environ.get("CONTROLFINDER_DB_URL", LOCAL_URL)
    server = connect(url)
    try:
        if not server.run("SELECT 1 FROM pg_database WHERE datname = :name", name=CHECK_DATABASE):
            try:
                server.run(f"CREATE DATABASE {CHECK_DATABASE}")
            except DatabaseError as exc:
                if not (exc.args and isinstance(exc.args[0], dict) and exc.args[0].get("C") == "42P04"):
                    raise  # 42P04: another run made it at the same moment, which is fine
    finally:
        server.close()
    return connect(urllib.parse.urlparse(url)._replace(path="/" + CHECK_DATABASE).geturl())


def versions(conn):
    postgres = conn.run("SHOW server_version")[0][0]
    pgvector = conn.run("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    return {"postgresql": postgres, "pgvector": pgvector[0][0] if pgvector else None}


def vector_text(values):
    """A vector in pgvector's text form. repr() keeps every digit, so float32 values arrive unchanged."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def create_table(conn):
    conn.run("CREATE EXTENSION IF NOT EXISTS vector")
    conn.run("DROP TABLE IF EXISTS controls")
    conn.run(f"""CREATE TABLE controls (
        position integer NOT NULL UNIQUE,  -- catalog order, for ties
        id text PRIMARY KEY,
        label text NOT NULL,
        title text NOT NULL,
        family text NOT NULL,
        parent text,
        statement text NOT NULL,
        guidance text NOT NULL,
        embedding vector({DIMENSIONS}) NOT NULL)""")


def load(conn, controls, vectors):
    """One transaction, so a failed load leaves no half-filled table."""
    conn.run("BEGIN")
    for position, (c, v) in enumerate(zip(controls, vectors, strict=True)):
        conn.run("""INSERT INTO controls (position, id, label, title, family, parent, statement, guidance, embedding)
                    VALUES (:position, :id, :label, :title, :family, :parent, :statement, :guidance, CAST(:embedding AS vector))""",
                 position=position, id=c.id, label=c.label, title=c.title, family=c.family, parent=c.parent,
                 statement=c.statement, guidance=c.guidance, embedding=vector_text(v))
    conn.run("COMMIT")
    conn.run("ANALYZE controls")


def ensure_loaded(conn, controls, vectors):
    """Creates and fills the table unless it already holds these controls in catalog order. True if it filled it.
    It checks the ids, not the vectors: the table is only ever filled from the committed embeddings."""
    conn.run("CREATE EXTENSION IF NOT EXISTS vector")
    if conn.run("SELECT to_regclass('controls') IS NOT NULL")[0][0]:
        if [row[0] for row in conn.run("SELECT id FROM controls ORDER BY position")] == [c.id for c in controls]:
            return False
    create_table(conn)
    load(conn, controls, vectors)
    return True


def create_index(conn):
    conn.run(f"DROP INDEX IF EXISTS {INDEX}")
    conn.run(f"""CREATE INDEX {INDEX} ON controls USING hnsw (embedding vector_ip_ops)
                 WITH (m = {HNSW['m']}, ef_construction = {HNSW['ef_construction']})""")


def use_index(conn, ef_search=None):
    """Index searches after this use the HNSW index. ef_search=None keeps pgvector's default (40)."""
    conn.run("SET enable_seqscan = off")  # on a table this small, the planner often skips the index
    if ef_search is None:
        conn.run("RESET hnsw.ef_search")
    else:
        conn.run(f"SET hnsw.ef_search = {int(ef_search)}")


def use_planner(conn):
    """Back to PostgreSQL's own choice of whether to use the index."""
    conn.run("RESET enable_seqscan")
    conn.run("RESET hnsw.ef_search")


def nearest(conn, vector, k, exact):
    """Ids of the k controls closest to the vector, closest first."""
    return [row[0] for row in conn.run(EXACT_SQL if exact else INDEX_SQL, v=vector_text(vector), k=k)]


def scans(conn, vector, k, exact):
    """The scans in the query plan, as (type, index name) pairs, to check which search really runs."""
    plan = conn.run("EXPLAIN (FORMAT JSON) " + (EXACT_SQL if exact else INDEX_SQL), v=vector_text(vector), k=k)[0][0]
    plan = json.loads(plan) if isinstance(plan, str) else plan
    found, stack = [], [plan[0]["Plan"]]
    while stack:
        node = stack.pop()
        if "Scan" in node["Node Type"]:
            found.append((node["Node Type"], node.get("Index Name")))
        stack.extend(node.get("Plans", []))
    return found
