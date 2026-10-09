# Plan

Control Finder answers security questions from NIST SP 800-53 with cited controls, says "not covered" when nothing fits, and measures all of it. The concept it teaches is retrieval done right: keyword and vector search, combining them, reranking, a vector database, and an API, each step measured against an answer key.

## Steps
1. **The answer key and a keyword baseline** (this step).
   - The answer key comes from NIST's own data: SP 800-171 Rev 3's 97 requirements and the 157 SP 800-53 controls they were derived from.
   - BM25 keyword search nearly aces it: MRR 1.000 on dev and 0.983 on test (`docs/results.md`). NIST adapted each requirement's wording from its source controls, so this key is easy for keyword search and can't separate better methods.
2. **A harder question set in plain English.** The next step, because step 1 showed the need.
   - First choice: NIST's official mapping from the Cybersecurity Framework (CSF) 2.0 to SP 800-53, if its download works. CSF outcomes are worded differently from controls, for example "Identities and credentials for authorized users, services, and hardware are managed by the organization".
   - Also: a small set of plain-English questions, each answer checked against the control text and labeled as hand-made, plus questions the catalog doesn't cover, to measure "not covered".
3. **Vector search and hybrid search.** Local embeddings (nomic-embed-text through Ollama, free), then keywords and vectors combined with reciprocal rank fusion. Also compared: searching whole controls against searching their numbered parts.
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
