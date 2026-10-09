# Control Finder

Ask a security question in plain English and get the NIST SP 800-53 controls that answer it, each one cited and linked. When the catalog doesn't cover a question, it says so. Its accuracy is measured against an answer key built from NIST's own data, and every result is published.

**Status:** in progress. Steps 1 and 2 of 7 are done: two answer keys from NIST's own data, and a keyword-search baseline on both. The steps are in [docs/plan.md](docs/plan.md).

## Results so far
Keyword search (BM25) over the 1,014 active controls and enhancements, scored on the test questions of two answer keys built from NIST's own data ([docs/results.md](docs/results.md)):

| Answer key | Test questions | MRR | First result right | A right control in the top 10 |
|---|---|---|---|---|
| SP 800-171 Rev 3 requirements → their source controls | 45 | 0.983 | 97.8% | 100% |
| CSF 2.0 outcomes → their mapped controls | 46 | 0.257 | 17.4% | 47.8% |

- **SP 800-171 is easy for keywords:** NIST adapted each requirement's wording from its source controls, so question and answer share many words.
- **CSF 2.0 is the real test:** its outcomes are worded unlike the controls ("Vulnerabilities in assets are identified, validated, and recorded"). The next steps (vector search, hybrid search, reranking) are measured on it.
- **Settings are chosen on dev questions only.** One exception, stated openly: a first exploratory run also printed CSF's test numbers. Nothing was tuned on them, and the rule that picks the search text (best dev MRR) was already fixed in code.

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py        # NIST's files, pinned and hash-checked
.venv/bin/python scripts/run_keyword_baseline.py # writes docs/results.md and both results files
.venv/bin/python scripts/build_csf_answer_key.py # optional: rebuilds answer_keys/csf2_sp800_53.json from NIST
.venv/bin/python -m pytest -q                    # 19 tests, including that the published numbers reproduce
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
