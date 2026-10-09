# Control Finder

Ask a security question in plain English and get the NIST SP 800-53 controls that answer it, each one cited and linked. When the catalog doesn't cover a question, it says so. Its accuracy is measured against an answer key built from NIST's own data, and every result is published.

**Status:** in progress. Step 1 of 7 is done: the answer key and a keyword-search baseline. The steps are in [docs/plan.md](docs/plan.md).

## Results so far
Keyword search (BM25) over the 1,014 active controls and enhancements, scored against NIST's own links ([docs/results.md](docs/results.md)):

| Questions | MRR | First result right | Right controls in the top 10 |
|---|---|---|---|
| 52 dev | 1.000 | 100% | 95.7% |
| 45 test | 0.983 | 97.8% | 100% |

Keyword search nearly aces this answer key, and that's the finding: NIST adapted each SP 800-171 requirement's wording from its source controls, so question and answer share many words. A harder, plain-English question set comes next, so the better methods (vectors, hybrid search, reranking) can be told apart.

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py        # NIST's files, pinned and hash-checked
.venv/bin/python scripts/run_keyword_baseline.py # writes docs/results.md and results/keyword_baseline.json
.venv/bin/python -m pytest -q                    # 14 tests, including that the published numbers reproduce
```

## The data
- **The catalog:** NIST SP 800-53 Rev 5.2.0, with the SP 800-53A Rev 5.2.0 assessment procedures, from NIST's machine-readable OSCAL files ([usnistgov/oscal-content](https://github.com/usnistgov/oscal-content)). It has 20 families, 300 active controls, 714 active enhancements, and 3,715 assessment objectives.
- **The answer key:** NIST SP 800-171 Rev 3, from the same repository. Each of its 97 active requirements links to the SP 800-53 controls it was derived from (157 links in all), so a search that returns those controls is right by NIST's own record.
- **License:** NIST's files are in the public domain in the United States, and NIST waives copyright worldwide under CC0 1.0. They're downloaded at a pinned commit and checked against SHA-256 hashes, never committed here.

## Notices
- Not affiliated with or endorsed by NIST.
- A reference tool, not legal or compliance advice. Check the cited official text.

## License
MIT for the code in this repository. NIST's data keeps its own public-domain terms.
