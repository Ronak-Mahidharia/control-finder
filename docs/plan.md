# Plan

Control Finder answers security questions from NIST SP 800-53 with cited controls, says "not covered" when nothing fits, and measures all of it. The concept it teaches is retrieval done right: keyword and vector search, combining them, reranking, a vector database, and an API, each step measured against an answer key.

## Steps
1. **The answer key and a keyword baseline** (done).
   - The answer key comes from NIST's own data: SP 800-171 Rev 3's 97 requirements and the 157 SP 800-53 controls they were derived from.
   - BM25 keyword search nearly aces it: MRR 1.000 on dev and 0.983 on test (`docs/results.md`). NIST adapted each requirement's wording from its source controls, so this key is easy for keyword search and can't separate better methods.
2. **A harder question set** (done). NIST's CSF 2.0 Reference Tool maps each of its 106 outcomes to SP 800-53 controls (731 in all). The outcomes are worded unlike controls, and keyword search drops to MRR 0.257 on test (from 0.983 on SP 800-171), so this key can tell methods apart.
   - Later, with step 6: a small set of plain-English questions, each answer checked against the control text and labeled as hand-made, plus questions the catalog doesn't cover, to measure "not covered".
   - Known limit: CI checks the committed CSF key, not NIST's spreadsheet itself (that check runs locally when the spreadsheet is downloaded).
3. **Vector search and hybrid search** (done). Local embeddings (nomic-embed-text through Ollama, free), and keywords plus vectors combined with reciprocal rank fusion (k = 60, not tuned).
   - On CSF 2.0 test: vector MRR 0.453 against keyword 0.257; a right control in the top 10 for 78.3% against 47.8%.
   - Hybrid found the most right controls on CSF 2.0 (recall@10 0.334) but put a right one first less often than vectors alone (MRR 0.411), and it kept most of keyword search's lead on SP 800-171 (0.967). Next: weighting the two lists, and reranking (step 4).
   - Not done yet: searching a control's numbered parts instead of the whole control.
4. **Reranking.** A cross-encoder reorders the top results; measured with and without.
5. **A vector database and an API.** PostgreSQL with pgvector, and a FastAPI service.
6. **Cited answers.** A local model writes the answer from the retrieved controls only, cites their IDs, and says "not covered" when nothing fits. Measured: citation precision, recall, and F1; faithfulness (every cited ID was retrieved); refusal accuracy; latency.
7. **An MCP server, a website, and a write-up.**

## Rules
- NIST's data only: public domain in the US and CC0 worldwide. Nothing from any employer, and nothing about contracts.
- Free: local models and free tiers only.
- Settings are chosen on dev questions only; test questions are scored once per method.
- Every number in the README comes from a script in this repository and is checked in CI.
- Labels: "Not affiliated with or endorsed by NIST"; a reference tool, not legal or compliance advice.
