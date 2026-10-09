# ============================================================================
# Downloads NIST's OSCAL files into data/raw/.
#
# Run:   python scripts/download_data.py
#   - Fetches each file from usnistgov/oscal-content at one pinned commit, so
#     every run reads exactly the same data.
#   - Checks each file's SHA-256 hash and stops if it doesn't match.
#   - Skips files already downloaded with the right hash.
# NIST's files are public domain in the US, plus CC0 1.0 worldwide (see the
# repository's LICENSE.md). They're never committed here.
# ============================================================================
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
COMMIT = "78650f02ad9321bb7b817846f8fbd4f2bcd620de"  # oscal-content main on Oct 8, 2026
FILES = {
    # SP 800-53 Rev 5.2.0 controls, with the SP 800-53A Rev 5.2.0 assessment procedures.
    "NIST_SP-800-53_rev5_catalog.json": (
        "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
        "01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062",
    ),
    # SP 800-171 Rev 3: its requirements link to the SP 800-53 controls they came from.
    "NIST_SP800-171_rev3_catalog.json": (
        "nist.gov/SP800-171/rev3/json/NIST_SP800-171_rev3_catalog.json",
        "21b6f3b118b6e5b305aaed3a0e4b70fa5d1d9aa388a0b92e7d6ddfed69e93ac3",
    ),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (path, expected) in FILES.items():
        target = RAW / name
        if target.exists() and sha256(target.read_bytes()) == expected:
            print(f"{name}: already here")
            continue
        url = f"https://raw.githubusercontent.com/usnistgov/oscal-content/{COMMIT}/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "control-finder (personal project)"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if sha256(data) != expected:
            sys.exit(f"{name}: SHA-256 doesn't match the pinned hash; not saved.")
        target.write_bytes(data)
        print(f"{name}: downloaded, {len(data):,} bytes, hash checked")


if __name__ == "__main__":
    main()
