# Control Finder

Ask a security question in plain English and get the NIST SP 800-53 controls that answer it, each one cited and linked. When the catalog doesn't cover a question, it says so. Its accuracy is measured against an answer key built from NIST's own data, and every result is published.

**Status:** in progress. Steps 1 to 4 of 7 are done: two answer keys from NIST's own data, and keyword, vector, hybrid, and reranked search measured on both. The steps are in [docs/plan.md](docs/plan.md).

## Results so far
Three ways to search the 1,014 active controls and enhancements, scored on the test questions of two answer keys built from NIST's own data. Settings were chosen on dev questions only; each method was scored once on test. Details: [keyword search](docs/results.md), [vector and hybrid search](docs/results-vector-hybrid.md), [weighting and reranking](docs/results-rerank.md).

| Answer key | Method | Test questions | MRR | First result right | A right control in the top 10 |
|---|---|---|---|---|---|
| CSF 2.0 | Keyword (BM25) | 46 | 0.257 | 17.4% | 47.8% |
| CSF 2.0 | Vector (nomic-embed-text) | 46 | 0.453 | 28.3% | 78.3% |
| CSF 2.0 | Hybrid (RRF) | 46 | 0.411 | 26.1% | 78.3% |
| CSF 2.0 | Hybrid, vector weight 3 | 46 | 0.463 | 32.6% | 76.1% |
| CSF 2.0 | Hybrid + reranker (mxbai-rerank-xsmall-v1) | 46 | 0.491 | 32.6% | 80.4% |
| SP 800-171 | Keyword (BM25) | 45 | 0.983 | 97.8% | 100.0% |
| SP 800-171 | Vector (nomic-embed-text) | 45 | 0.902 | 84.4% | 97.8% |
| SP 800-171 | Hybrid (RRF) | 45 | 0.967 | 95.6% | 97.8% |
| SP 800-171 | Hybrid, vector weight 3 | 45 | 0.906 | 84.4% | 97.8% |
| SP 800-171 | Hybrid + reranker (mxbai-rerank-xsmall-v1) | 45 | 0.924 | 88.9% | 97.8% |

- **CSF 2.0 is the real test.** Its outcomes are worded unlike the controls ("Vulnerabilities in assets are identified, validated, and recorded"). Searching by meaning nearly doubles keyword search's MRR there.
- **Hybrid search isn't automatically better.** On CSF 2.0 it found the most right controls (recall@10 0.334, against 0.303 for vectors alone) but put a right one first less often. Across both keys it's the most balanced: on SP 800-171, which keyword search nearly aces because NIST adapted each requirement's wording from its source controls, it keeps most of keyword search's lead.
- **Reranking helps most on the hard key.** Weighting the vector list 3 times (chosen on dev) and reranking its top 30 with mxbai-rerank-xsmall-v1 gives CSF 2.0's best MRR so far, 0.491. A bigger reranker isn't automatically better: bge-reranker-base did worse on dev than no reranking (0.405 against 0.470). And there's a trade-off: tuned for CSF 2.0, the weighted search loses ground on SP 800-171. Reranking can only reorder the top 30, which hold 53.2% of CSF 2.0's right controls on test.
- **Everything runs locally and free:** the embedding model is nomic-embed-text through Ollama. Every question's top 100 vector results and every reranker score are committed, so CI checks these numbers without the models.
- **Stated openly:** a first exploratory run in step 2 also printed CSF's keyword test numbers. Nothing was tuned on them.

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py        # NIST's files, pinned and hash-checked
.venv/bin/python scripts/run_keyword_baseline.py # writes docs/results.md and both results files
.venv/bin/python scripts/build_csf_answer_key.py # optional: rebuilds answer_keys/csf2_sp800_53.json from NIST
.venv/bin/python scripts/run_vector_and_hybrid.py # needs Ollama: ollama pull nomic-embed-text (about 30 s)
.venv/bin/pip install -r requirements-rerank.txt   # optional: PyTorch and sentence-transformers
.venv/bin/python scripts/run_rerank.py            # tries 3 rerankers on dev, runs the best on test (about 8 min on CPU)
.venv/bin/python -m pytest -q                    # 33 tests, including that the published numbers reproduce
```

## The data
- **The catalog:** NIST SP 800-53 Rev 5.2.0, with the SP 800-53A Rev 5.2.0 assessment procedures, from NIST's machine-readable OSCAL files ([usnistgov/oscal-content](https://github.com/usnistgov/oscal-content)). It has 20 families, 300 active controls, 714 active enhancements, and 3,715 assessment objectives.
- **Answer key 1:** NIST SP 800-171 Rev 3, from the same repository. Each of its 97 active requirements links to the SP 800-53 controls it was derived from (157 links in all), so a search that returns those controls is right by NIST's own record.
- **Answer key 2:** NIST's Cybersecurity Framework (CSF) 2.0 Reference Tool. Each of its 106 outcomes lists the SP 800-53 controls NIST maps it to (731 in all). Only NIST's outcome text and the SP 800-53 IDs are kept, in `answer_keys/csf2_sp800_53.json`. NIST builds the spreadsheet fresh on each download, so its data parts, not the whole file, are checked against a SHA-256 hash.
- **License:** NIST's files are in the public domain in the United States, and NIST waives copyright worldwide under CC0 1.0. They're downloaded at a pinned commit and checked against SHA-256 hashes, never committed here.

## Notices
- Not affiliated with or endorsed by NIST.
- A reference tool, not legal or compliance advice. Check the cited official text.

## License
MIT for the code in this repository. NIST's data keeps its own public-domain terms.
