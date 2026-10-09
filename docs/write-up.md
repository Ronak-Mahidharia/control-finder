# Building Control Finder: what I measured, and what I learned

Control Finder takes a security question in plain English, such as "We store customer passwords. How should they be protected?", and answers with the NIST SP 800-53 controls that apply, each one cited. When the catalog doesn't cover a question, it says so. It has six parts: two answer keys built from NIST's own data, a search that combines keywords and meaning, a vector database and an API, cited answers from a local model, a server that lets AI assistants use it, and a website that runs on your computer.

I set myself one rule: no claim without a number behind it. This write-up follows the project in order, with the numbers that shaped each decision. The details are in the [README](../README.md) and the documents it links.

## An answer key that was too easy
SP 800-53 is NIST's catalog of security controls: 1,014 active controls and enhancements in Rev 5.2.0. To measure a search over it, I needed questions with known right answers, and NIST had already written some.
- **The first key:** NIST SP 800-171 Rev 3 lists, for each of its 97 active requirements, the SP 800-53 controls it was derived from, 157 links in all. A requirement is the question, and its source controls are the right answers.
- **Keyword search nearly aced it.** BM25 scored an MRR of 0.983 on the test questions and put a right control first for 97.8% of them. NIST adapted each requirement's wording from its source controls, so matching words was enough. A near-perfect score said more about the key than about the search.

## A harder key from NIST's CSF 2.0
NIST's Cybersecurity Framework (CSF) 2.0 Reference Tool maps each of its 106 outcomes to SP 800-53 controls, 731 mappings in all. The outcomes are worded unlike the controls ("Vulnerabilities in assets are identified, validated, and recorded"), so this key can tell search methods apart.
- Keyword search dropped to an MRR of 0.257. It put a right control first for 17.4% of the test questions, and in the top 10 for 47.8%.
- NIST builds its spreadsheet fresh on every download, with new timestamps, so the file's hash changes each time. Only its data parts are hashed, and they stayed the same across downloads.
- From here on, every setting was chosen on dev questions only, and each method was scored once on test.

## Searching by meaning, then combining
| Method | CSF 2.0 MRR | CSF 2.0: a right control in the top 10 | SP 800-171 MRR |
|---|---|---|---|
| Keyword (BM25) | 0.257 | 47.8% | 0.983 |
| Vector (nomic-embed-text, local) | 0.453 | 78.3% | 0.902 |
| Hybrid (reciprocal rank fusion) | 0.411 | 78.3% | 0.967 |
| Hybrid, the vector list weighted 3 times | 0.463 | 76.1% | 0.906 |
| Weighted hybrid, top 30 reranked | **0.491** | 80.4% | 0.924 |

- **Meaning beat keywords on the hard key:** vector search nearly doubled keyword search's MRR there. On the easy key, keywords stayed ahead.
- **Combining wasn't automatically better.** Equal-weight fusion found the most right controls on CSF 2.0 (recall@10 0.334, against 0.303 for vectors alone) but put a right one first less often. Weighting the vector list 3 times, chosen on dev from 1, 1.5, 2, and 3, helped on CSF 2.0 and cost ground on SP 800-171: a trade-off, not a free win.
- **A bigger reranker wasn't better.** Of three cross-encoders tried on dev, the smallest, mxbai-rerank-xsmall-v1, scored best (MRR 0.596), and bge-reranker-base did worse than no reranking (0.405 against 0.470).
- **Reranking has a ceiling:** the top 30 candidates held 53.2% of CSF 2.0's right controls on test, so only a better first search can raise it.

## A real vector database, and two settings that are easy to miss
I loaded the controls into PostgreSQL with pgvector and checked its search against the in-memory one.
- **Exact search matched.** It gave the same top 10 for all 203 questions, so every published score held. In 6 of the 203 top-100 lists, two near-tied neighbors swapped places, which is float32 rounding.
- **pgvector's default returns only 40 rows.** Its HNSW index keeps 40 candidates by default (`hnsw.ef_search`) and returns no more, so a request for a top 100 quietly gets 40. Over 10 index builds, that changed the step 3 hybrid's top 10 for 60 to 63 of the 203 questions. With ef_search set to 100, every test score matched exact search.
- **PostgreSQL decides when to use the index.** With the index in place, it used it for a top 10, which made that result approximate, and scanned the whole table for a top 100. My first "exact" query was caught by this, so exact search now orders by pgvector's `inner_product()`, which no index can serve.
- **At this size, the database is the slow part.** Exact search took 4.07 ms per question (median) and the forced index 3.25 ms, while the same search in memory took 0.08 ms. Sending the question alone took 1.76 ms. At 1,014 controls, the database is there to share the data, not for speed.

## An API that returns what was measured
A FastAPI service answers `/search`, `/answer`, and `/controls/{id}`. Run for real, with the model embedding each question, it gave the published top 10 for all 203 questions with every method. A hybrid search took 22.79 ms when the question was new, 11.31 ms of it embedding the question, and 5.37 ms when it had been asked before. Keyword search keeps working when the database or the model is down, and the other methods answer 503 with the reason.

## Cited answers, and a model that answered anyway
For answers, a local model reads the search's top 10 controls and answers from them only. Its JSON must follow a schema that allows only the shown controls' labels as citations, so it can't cite a control it wasn't shown. Besides NIST's keys, I wrote two small question sets by hand, labeled as such: 32 plain-English questions, each with anchor controls that plainly address it, and 40 questions the catalog doesn't answer.

I compared two local models, two prompt wordings, and 5 or 10 shown controls on dev questions. qwen3:8b with the first wording and 10 controls won. It said "not covered" to 100% of the out-of-scope dev questions, against 45% to 60% for granite4:3b, which was about twice as fast. On the test questions, run once:
- **Citations:** F1 0.482 (precision 0.474, recall 0.586) on NIST's keys, against 0.420 for citing the search's top result with no model. It did far better on SP 800-171 (0.770) than on CSF 2.0 (0.201), whose outcomes each map to many controls.
- **"Not covered":** for 19 of 20 out-of-scope questions, and for none of the 110 covered ones. The one it got wrong: asked whether California allows recording a call without telling the other person, it answered from memory and cited two controls that don't apply.
- **Plain-English questions:** it cited an anchor for 78.9%. All 4 misses cited an enhancement of an anchor instead, such as AC-2(3) "Disable Accounts" for AC-2.

Reading every answer turned up the most useful finding of the project. When the model said a question wasn't covered, its own text often answered the question anyway, from memory: "443" for the HTTPS port, the capital of Australia, a Python one-liner. That happened in 9 of its 19 "not covered" answers on test. A schema can keep citations honest, but not free text. So the API never shows that text: a "not covered" answer always gets the same fixed sentence. I added this check after reading the test answers, and it changed no score or choice.

The main weakness is citing too much: 4.15 controls per answer on NIST's test questions, which kept precision below half.

## Letting AI assistants use it
An MCP server gives Claude Desktop, Claude Code, and other MCP clients three read-only tools: `search_controls`, `get_control`, and `answer_question`. It searches in memory, which the database step showed gives the same rankings, so it needs only Ollama. Over stdio, with the model embedding each question live, it gave the published top 20 for all 203 questions, connected in 336 ms, and answered later searches in 12.7 ms. Its instructions ask the assistant to cite controls by label, describe only what a control's text says, and pass on "not covered" instead of answering from general knowledge.

## Seeing it work: a local website
A small React and TypeScript website, served by the API at http://127.0.0.1:8000/, puts it all on one page. You type a question, see the closest controls, open any control's text and what an assessor checks, or ask for a cited answer. One script starts everything. A browser check clicks through the site in headless Chrome on a desktop and a phone, in light and dark mode, and all 25 of its checks pass. Every color pair meets WCAG AA contrast in both themes, and the page may only talk to its own API.

Looking at the screenshots caught a bug the tests had missed: the line listing the controls the model saw showed ids like "ia-5.1" instead of NIST's labels like "IA-5(1)". The API sends ids there, but my test data had used labels. The test data now matches the API, and the browser check looks for the labels.

Trying the site by hand found a second problem: the "Search by" choice seemed to do nothing. It applies only to Find controls, because answers always use hybrid search, the setup they were measured with. And changing it left the earlier results on the page with nothing to say they were stale. The page now says which search found every list and answer, notes under the choice that answers always use hybrid, and asks for a new search when the choice changes.

## What I'd do next
- **Cite less.** Ask for the single best control first, or rerank before answering, and measure whether precision rises without losing recall.
- **Check the sentences, not just the citations.** Here, faithful means every citation was shown to the model. It doesn't yet check that each sentence is supported by the cited text.
- **Count an anchor's enhancements** in the plain-English set, since all 4 misses cited one.
- **Search a control's numbered parts** instead of the whole control, so a long control's main point doesn't get diluted.

## How it's checked
- 110 Python tests run on every change, with the website's own tests, in GitHub Actions pinned to exact commits, with read-only permissions. The 11 database tests run against a pgvector container pinned by digest.
- The embeddings, the vector lists, every reranker score, and every model answer are committed, so CI recomputes every published number without running a model.
- Each step planted bugs on purpose to test the tests, 79 in all, and every one was caught; one was caught only after a test was added for it.
- All models run locally through Ollama, so the project costs $0 to run, and it uses only NIST's public-domain data.
