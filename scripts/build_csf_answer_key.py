# ============================================================================
# Builds the second answer key from NIST's Cybersecurity Framework (CSF) 2.0.
#
# Run:   python scripts/build_csf_answer_key.py
#   1. Downloads the CSF 2.0 Reference Tool's spreadsheet from NIST and checks
#      the SHA-256 hash of its data (NIST's live export, so a change is reported, not used).
#   2. For each CSF 2.0 outcome (subcategory), reads the SP 800-53 Rev 5.1.1
#      controls NIST lists in its "Informative References" column.
#   3. Writes answer_keys/csf2_sp800_53.json: 106 outcomes and their controls.
# The spreadsheet stays in data/raw/ (never committed). The small answer key is
# committed, so tests and CI don't depend on NIST's download page. Only NIST's
# own text and SP 800-53 IDs are kept; other frameworks' references are not.
# CSF outcomes are worded unlike controls ("Vulnerabilities in assets are
# identified, validated, and recorded"), which makes this key the hard one.
# ============================================================================
import hashlib
import html
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "NIST_CSF_2.0_reference_tool.xlsx"
OUT = ROOT / "answer_keys" / "csf2_sp800_53.json"
URL = "https://csrc.nist.gov/extensions/nudp/services/json/csf/download?olirids=all"
# NIST builds this file on each download and stamps it with the time, so the whole file's hash changes.
# The hash is taken over the parts that hold the data: the "CSF 2.0" sheet and its shared text.
DATA_SHA256 = "5c34a08c7a8dbd40ae6f594581d63769fde2fbf3e941f353716b89e047234c59"  # checked Oct 8, 2026 (two downloads, same data)

SUBCATEGORY = re.compile(r"^([A-Z]{2}\.[A-Z]{2}-\d{2}):\s*(.+)$", re.S)
REFERENCE = re.compile(r"SP 800-53 Rev 5\.1\.1:\s*([A-Z]{2})-(\d{2})(?:\((\d{2})\))?")
CELL = re.compile(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', re.S)  # also matches empty "<c .../>" cells


def download():
    if not RAW.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "control-finder (personal project)"})
        with urllib.request.urlopen(request, timeout=60) as response:
            RAW.parent.mkdir(parents=True, exist_ok=True)
            RAW.write_bytes(response.read())
    book = zipfile.ZipFile(RAW)
    digest = hashlib.sha256(book.read("xl/sharedStrings.xml") + book.read("xl/worksheets/sheet2.xml")).hexdigest()
    if digest != DATA_SHA256:
        sys.exit(f"NIST's CSF 2.0 data changed (SHA-256 {digest}). Check it, then update DATA_SHA256 on purpose.")


def rows(path):
    """The 'CSF 2.0' sheet as a list of {column letter: text}."""
    book = zipfile.ZipFile(path)
    shared = [html.unescape(re.sub(r"<[^>]+>", "", s))
              for s in re.findall(r"<si>(.*?)</si>", book.read("xl/sharedStrings.xml").decode("utf-8"), re.S)]
    names = re.findall(r'<sheet [^>]*name="([^"]+)"', book.read("xl/workbook.xml").decode("utf-8"))
    sheet = book.read(f"xl/worksheets/sheet{names.index('CSF 2.0') + 1}.xml").decode("utf-8")
    out = []
    for row in re.findall(r"<row [^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for column, attrs, body in CELL.findall(row):
            value = re.search(r"<v>(.*?)</v>", body or "", re.S)
            if value:
                cells[column] = shared[int(value.group(1))] if 't="s"' in attrs else html.unescape(value.group(1))
        out.append(cells)
    return out


def control_id(family, number, enhancement):
    return f"{family.lower()}-{int(number)}" + (f".{int(enhancement)}" if enhancement else "")


def build(sheet_rows):
    """Column C names the outcome; its references are in column E, on its row and the example rows below it."""
    outcomes, current = {}, None
    for row in sheet_rows:
        match = SUBCATEGORY.match(row.get("C", ""))
        if match:
            current = match.group(1)
            outcomes[current] = {"id": current, "text": " ".join(match.group(2).split()), "expected": set()}
        elif row.get("A") or row.get("B"):
            current = None  # a new function or category starts
        if current:
            outcomes[current]["expected"].update(control_id(*m) for m in REFERENCE.findall(row.get("E", "")))
    # Withdrawn CSF 1.1 outcomes ("[Withdrawn: ...]") have no SP 800-53 references and drop out here.
    return [{**o, "expected": sorted(o["expected"])} for o in outcomes.values() if o["expected"]]


def main():
    download()
    questions = build(rows(RAW))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "source": {"url": URL, "data_sha256": DATA_SHA256, "downloaded": "2026-10-08",
                   "note": "NIST CSF 2.0 Reference Tool export; only its SP 800-53 Rev 5.1.1 references are kept"},
        "questions": questions,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"{len(questions)} CSF 2.0 outcomes, {sum(len(q['expected']) for q in questions)} SP 800-53 references -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
