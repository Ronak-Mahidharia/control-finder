# Control Finder

Ask a security question in plain English and get the NIST SP 800-53 controls that answer it, each one cited, with NIST's text a click away. When the catalog doesn't cover a question, it says so. Its accuracy is measured against an answer key built from NIST's own data, and every result is published.

**Try it:** on your computer, with one script ([how](#try-it-on-your-computer)). There's no hosted demo, because the website needs the database and the local models.

[![CI](https://github.com/Ronak-Mahidharia/control-finder/actions/workflows/ci.yml/badge.svg)](https://github.com/Ronak-Mahidharia/control-finder/actions/workflows/ci.yml)

**What's here:**
- **Two answer keys from NIST's own data,** with keyword, vector, hybrid, and reranked search measured on both.
- **A vector database and an API** (PostgreSQL with pgvector, and FastAPI), both checked against the published rankings.
- **Cited answers** from a local model that says when a question isn't covered.
- **An MCP server** that lets AI assistants use the same search.
- **A website** that runs on your computer.

How it was built is in [docs/plan.md](docs/plan.md), and the story, with the numbers behind each decision, is in [the write-up](docs/write-up.md).

## Try it on your computer
```bash
scripts/start_local.sh   # starts the database, checks Ollama, builds the website once, and opens http://127.0.0.1:8000/
scripts/stop_local.sh    # stops it again
```
The first time, you'll need the Python setup in [Run it](#run-it), Docker Desktop, Node 24, and Ollama with `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`. Everything listens on 127.0.0.1 only, so nothing outside your computer can reach it.

![The answer to "We store customer passwords. How should they be protected?": covered, citing IA-5(1) and IA-5(18), with the 10 controls the model was shown](docs/screenshots/answer-desktop-light.png)

- Type a question or pick an example. **Find controls** lists the 10 closest controls, each with NIST's text and what an assessor checks. **Answer with citations** gives the local model's answer, citing only controls it was shown.
- **Search by** picks hybrid, meaning-only, or keyword search for Find controls, and every list says which one found it. Answers always use hybrid search, the setup they were measured with, and the page says so.
- A question the catalog doesn't cover, such as the HTTPS port, gets "Not covered" and a fixed sentence.
- A browser check (`node scripts/check_website.mjs`) clicks through the site in headless Chrome on a desktop and a phone, in light and dark mode: 27 of 27 checks pass. More pictures: [search on a desktop](docs/screenshots/search-desktop-light.png), [an answer in dark mode](docs/screenshots/answer-desktop-dark.png), [search on a phone](docs/screenshots/search-phone-light.png).

## How it fits together
```mermaid
flowchart TB
    nist[("NIST's files<br/>SP 800-53 with 800-53A, SP 800-171, CSF 2.0")]

    subgraph offline["Built and measured offline, in Python"]
        keys["Answer keys<br/>NIST's own links, plus hand-made questions"]
        emb["Embeddings of every active control<br/>committed in embeddings/"]
        exp["Experiments<br/>keyword, vector, hybrid, reranking, cited answers"]
        results["Published results<br/>results/ and docs/"]
    end

    subgraph live["Used live, on this computer only"]
        db[("PostgreSQL with pgvector<br/>in Docker")]
        api["API (FastAPI) on 127.0.0.1<br/>/search, /answer, /controls, /health"]
        site["Website<br/>React and TypeScript"]
        mcp["MCP server<br/>3 read-only tools over stdio"]
        assistant["AI assistant<br/>Claude Desktop, Claude Code"]
        ollama["Ollama<br/>nomic-embed-text, qwen3:8b"]
    end

    nist --> keys & emb
    keys & emb --> exp --> results
    emb --> db --> api --> site
    emb --> mcp
    mcp <--> assistant
    api & mcp --> ollama
```

- **Offline, in Python:** NIST's own links between its documents are the answer key. Settings are chosen on dev questions only, and every result is published in [`results/`](results) and [`docs/`](docs).
- **Live:** the API searches the database (exact search, so it returns the rankings that were measured) and serves the website. The MCP server searches the same embeddings in memory, which gives the same rankings with no database. Questions go only to Ollama on this computer.

## Results
Three ways to search the 1,014 active controls and enhancements, scored on the test questions of two answer keys built from NIST's own data. Settings were chosen on dev questions only; each method was scored once on test. Details: [keyword search](docs/results.md), [vector and hybrid search](docs/results-vector-hybrid.md), [weighting and reranking](docs/results-rerank.md), [following NIST's links](docs/results-related.md).

| Answer key | Method | Test questions | MRR | First result right | A right control in the top 10 |
|---|---|---|---|---|---|
| CSF 2.0 | Keyword (BM25) | 46 | 0.257 | 17.4% | 47.8% |
| CSF 2.0 | Vector (nomic-embed-text) | 46 | 0.453 | 28.3% | 78.3% |
| CSF 2.0 | Hybrid (RRF) | 46 | 0.411 | 26.1% | 78.3% |
| CSF 2.0 | Hybrid, vector weight 3 | 46 | 0.463 | 32.6% | 76.1% |
| CSF 2.0 | Hybrid + reranker (mxbai-rerank-xsmall-v1) | 46 | 0.491 | 32.6% | 80.4% |
| CSF 2.0 | Hybrid + NIST's links between controls | 46 | 0.567 | 41.3% | 87.0% |
| SP 800-171 | Keyword (BM25) | 45 | 0.983 | 97.8% | 100.0% |
| SP 800-171 | Vector (nomic-embed-text) | 45 | 0.902 | 84.4% | 97.8% |
| SP 800-171 | Hybrid (RRF) | 45 | 0.967 | 95.6% | 97.8% |
| SP 800-171 | Hybrid, vector weight 3 | 45 | 0.906 | 84.4% | 97.8% |
| SP 800-171 | Hybrid + reranker (mxbai-rerank-xsmall-v1) | 45 | 0.924 | 88.9% | 97.8% |
| SP 800-171 | Hybrid + NIST's links between controls | 45 | 0.569 | 35.6% | 97.8% |

- **CSF 2.0 is the real test.** Its outcomes are worded unlike the controls ("Vulnerabilities in assets are identified, validated, and recorded"). Searching by meaning nearly doubles keyword search's MRR there.
- **Hybrid search isn't automatically better.** On CSF 2.0 it found the most right controls (recall@10 0.334, against 0.303 for vectors alone) but put a right one first less often. Across both keys it's the most balanced: on SP 800-171, which keyword search nearly aces because NIST adapted each requirement's wording from its source controls, it keeps most of keyword search's lead.
- **Reranking helps most on the hard key.** Weighting the vector list 3 times (chosen on dev) and reranking its top 30 with mxbai-rerank-xsmall-v1 gives CSF 2.0's best MRR, 0.491. A bigger reranker isn't automatically better: bge-reranker-base did worse on dev than no reranking (0.405 against 0.470). And there's a trade-off: tuned for CSF 2.0, the weighted search loses ground on SP 800-171. Reranking can only reorder the top 30, which hold 53.2% of CSF 2.0's right controls on test.
- **Following NIST's own links helps the hard key and hurts the easy one.** Each control lists related controls (3,509 links between the active ones); letting the hybrid search's top 10 lend a quarter of their score to the controls they list (chosen on CSF 2.0 dev from 12 settings) gives CSF 2.0's best MRR yet, 0.567, with a right control in the top 10 for 87.0% of questions. On SP 800-171 MRR falls from 0.906 to 0.569: its requirements come from one or two specific controls, and related ones push them down. So the API doesn't use it.
- **Everything runs locally and free:** the embedding model is nomic-embed-text through Ollama. The embeddings, every question's top 100 vector results, and every reranker score are committed, so CI checks these numbers without the models.
- **Stated openly:** a first exploratory run, while the CSF 2.0 key was being built, also printed its keyword test numbers. Nothing was tuned on them.

## The vector database
The controls are also loaded into PostgreSQL with pgvector, from the committed embeddings, and searched there. Details: [the vector database](docs/results-database.md).
- **Same results as before.** Exact search in the database returned the same top 10 as the in-memory search for all 203 questions, and every test score above came back. In 6 of the 203 top-100 lists, two near-tied neighbors swapped places (float32 rounding).
- **pgvector's default setting returns only 40 rows.** Its HNSW index keeps 40 candidates by default (`hnsw.ef_search`) and returns no more, so a request for a top 100 gets 40, with no error. Over 10 index builds, that changed the top 10 of hybrid search (RRF) for 60 to 63 of the 203 questions, and on CSF 2.0 test it put a right control in the top 10 for 76.1% of questions instead of 78.3%. With ef_search 100, every test score matched exact search.
- **At 1,014 controls, the index saves little.** For a top 100, PostgreSQL's planner skipped the index and scanned the table. Forced, the index took 3.25 ms per question (median) against 4.07 ms for exact search. Sending the question alone took 1.76 ms, and the in-memory search took 0.08 ms. These times come from one Apple-silicon Mac, so they're a guide, not a benchmark.

## The API
`GET /search` answers a question with the controls that fit it (`method` = hybrid, the default, vector, or keyword), `GET /controls/{id}` returns a control's text and its assessment objectives, and `GET /health` says what can be reached. Details: [the API](docs/results-api.md).
- **It returns the published rankings.** Run for real, with the model embedding each question, it gave the published top 10 for all 203 questions with every method, and hybrid's top 30 matched the reranker's candidates.
- **Speed:** a hybrid search took 22.79 ms (median) when the question was new, 11.31 ms of it embedding the question, and 5.37 ms when it had been asked before. Keyword search took 1.05 ms. These times come from one Apple-silicon Mac.
- **When something is down:** keyword search still works without the database or the model, and the other methods answer 503 with the reason. Questions are capped at 2,000 characters and results at 50.
- **Not served:** the reranker, which would add PyTorch to the service. The API serves the weighted hybrid search.

## Cited answers
`GET /answer` has a local model (qwen3:8b, chosen on dev) read the hybrid search's top 10 controls and answer from them only. Its citations can only name controls it was shown. Details: [cited answers](docs/results-answers.md).
- **On NIST's answer keys** (91 test questions), its citations scored F1 0.482 (precision 0.474, recall 0.586), against 0.420 for citing the search's top result with no model. The top 10 held 62.9% of NIST's links, a ceiling for recall. It did far better on SP 800-171 (F1 0.770) than on CSF 2.0 (0.201), whose outcomes each map to many controls.
- **It says when the catalog doesn't cover a question:** 95.0% of the hand-made out-of-scope test questions (100.0% of the unrelated ones and 88.9% of the security-adjacent ones), and it said so to 0 of 110 covered ones. The one it got wrong: asked whether California allows recording a call without telling the other person, it answered from memory and cited two controls that don't apply.
- **Its own "not covered" text can't be trusted.** In 9 of its 19 "not covered" answers on test, the model's text still answered the question from what it already knew (for example, the HTTPS port). So the API shows a fixed sentence instead.
- **Plain-English questions** (hand-made, 19 on test): it cited an anchor control for 78.9%. All 4 misses cited an enhancement of an anchor instead.
- **Two models compared on dev:** granite4:3b answered about twice as fast but said "not covered" to only 45% to 60% of the out-of-scope questions, against 100% for qwen3:8b. With qwen3:8b, an answer took 4.33 s (median, test) on one Apple-silicon Mac.

## The MCP server
AI assistants such as Claude Desktop and Claude Code can use Control Finder as three read-only tools: `search_controls`, `get_control`, and `answer_question`. Setup: [docs/mcp.md](docs/mcp.md). Results: [the MCP server](docs/results-mcp.md).
- **The published rankings, with no database.** It searches in memory, which gives the same rankings as the database. Run over stdio, with the model embedding each question live, it gave the published top 20 for 203 of 203 questions.
- **Speed:** it connected in 336 ms, and searches after the first took 12.7 ms (median), on one Apple-silicon Mac.
- **Its instructions** ask the assistant to cite controls by label, describe only what a control's text says, and pass on "not covered" instead of answering from general knowledge.

## Run it
Needs Python 3.12 or newer (`python3 --version` shows yours).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py        # NIST's files, pinned and hash-checked
.venv/bin/python scripts/run_keyword_baseline.py # writes docs/results.md and both results files
.venv/bin/python scripts/build_csf_answer_key.py # optional: rebuilds answer_keys/csf2_sp800_53.json from NIST
.venv/bin/python scripts/run_vector_and_hybrid.py # needs Ollama: ollama pull nomic-embed-text (about 30 s)
.venv/bin/pip install -r requirements-rerank.txt   # optional: PyTorch and sentence-transformers
.venv/bin/python scripts/run_rerank.py            # tries 3 rerankers on dev, runs the best on test (about 8 min on CPU)
docker compose up -d --wait                       # the database: PostgreSQL with pgvector, on 127.0.0.1:55432 only
.venv/bin/python scripts/run_database_check.py    # loads the controls and compares the searches (about 1 min)
.venv/bin/python -m uvicorn controlfinder.api:app --app-dir src --host 127.0.0.1 --port 8000  # the API (needs Ollama); docs at /docs
.venv/bin/python scripts/run_api_check.py         # runs the API and checks it against the published rankings (about 15 s)
.venv/bin/python scripts/run_answer_eval.py       # cited answers: needs qwen3:8b and granite4:3b in Ollama (about 55 min; answers are cached)
.venv/bin/pip install -e . --no-deps              # lets an AI assistant start the MCP server from any folder (docs/mcp.md)
.venv/bin/python scripts/run_mcp_check.py         # runs the MCP server over stdio and checks it (about 20 s)
(cd web && npm ci && npm test && npm run build)   # the website: its tests, then the build the API serves
node scripts/check_website.mjs                    # with scripts/start_local.sh running: the browser check, and the screenshots
.venv/bin/python -m pytest -q                    # 123 tests, including that the published numbers reproduce
CONTROLFINDER_DB_URL=postgresql://controlfinder:controlfinder-local@127.0.0.1:55432/controlfinder \
  .venv/bin/python -m pytest -q tests/test_database.py tests/test_api_database.py  # the 12 tests that need that database
```

The database tests and `scripts/run_database_check.py` rebuild their table in a database of their own on the same server (`controlfinder_check`), so the site's table is never touched and they can run while the site does.

## Project structure
| Folder | What's in it |
|---|---|
| [`src/controlfinder/`](src/controlfinder) | The Python package: the catalog, answer keys, search, database, API, cited answers, and MCP server |
| [`scripts/`](scripts) | One command per measurement, plus the start and stop scripts |
| [`web/`](web) | The website (React, TypeScript, Vite) and its tests |
| [`tests/`](tests) | The Python tests (pytest), including that every published number reproduces |
| [`answer_keys/`](answer_keys) | The CSF 2.0 key built from NIST's spreadsheet, and the two hand-made sets |
| [`embeddings/`](embeddings) | The embedding of every control and question, with SHA-256 hashes |
| [`results/`](results) and [`docs/`](docs) | Every result, as JSON and as reports, plus the write-up and the MCP setup |

## The data
- **The catalog:** NIST SP 800-53 Rev 5.2.0, with the SP 800-53A Rev 5.2.0 assessment procedures, from NIST's machine-readable OSCAL files ([usnistgov/oscal-content](https://github.com/usnistgov/oscal-content)). It has 20 families, 300 active controls, 714 active enhancements, and 3,715 assessment objectives.
- **Answer key 1:** NIST SP 800-171 Rev 3, from the same repository. Each of its 97 active requirements links to the SP 800-53 controls it was derived from (157 links in all), so a search that returns those controls is right by NIST's own record.
- **Answer key 2:** NIST's Cybersecurity Framework (CSF) 2.0 Reference Tool. Each of its 106 outcomes lists the SP 800-53 controls NIST maps it to (731 in all). Only NIST's outcome text and the SP 800-53 IDs are kept, in `answer_keys/csf2_sp800_53.json`. NIST builds the spreadsheet fresh on each download, so its data parts, not the whole file, are checked against a SHA-256 hash.
- **Hand-made sets:** 32 plain-English questions, each with anchor controls that plainly address it, and 40 questions SP 800-53 doesn't answer, in `answer_keys/`. Each file says it's hand-made and how it was made.
- **Embeddings:** `embeddings/` holds the 768 numbers nomic-embed-text gave each control and question (3.8 MB, with SHA-256 hashes in its manifest), so the database and CI don't need the model.
- **License:** NIST's files are in the public domain in the United States, and NIST waives copyright worldwide under CC0 1.0. They're downloaded at a pinned commit and checked against SHA-256 hashes, never committed here.

## Limits
- **No hosted demo.** The website needs the database and the local models, so it runs on your computer. The screenshots above show it.
- **Broad questions are hard.** On CSF 2.0, whose outcomes each map to many controls, a right control comes first far less often than on SP 800-171 (see [Results](#results)).
- **Answers are checked through their citations.** Every citation names a control the model was shown, but a sentence isn't yet checked against the text it cites.
- **One version of the catalog:** SP 800-53 Rev 5.2.0. A new release would need its files downloaded, the embeddings rebuilt, and everything measured again.

## Notices
- Not affiliated with or endorsed by NIST.
- A reference tool, not legal or compliance advice. Check the cited official text.

## License
MIT for the code in this repository. NIST's data keeps its own public-domain terms.
