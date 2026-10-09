# The website's published claims must match results/website.json (the browser check, run by
# scripts/check_website.mjs against the running site), and the pieces the README points to must exist.
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "website.json").read_text())


def test_every_browser_check_passed_and_its_screenshots_exist():
    assert RESULTS["passed"] == RESULTS["total"] >= 20
    assert all(c["passed"] for c in RESULTS["checks"]) and len(RESULTS["checks"]) == RESULTS["total"]
    for name in RESULTS["screenshots"]:
        assert (ROOT / "docs" / "screenshots" / f"{name}.png").stat().st_size > 10_000, name


def test_the_readme_and_write_up_report_the_check():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Try it on your computer")
    section = readme[start:readme.index("\n## ", start)]
    assert f"{RESULTS['passed']} of {RESULTS['total']} checks pass" in section
    for picture in re.findall(r"\((docs/screenshots/[\w-]+\.png)\)", section):
        assert (ROOT / picture).exists(), picture
    assert f"all {RESULTS['total']} of its checks pass" in (ROOT / "docs" / "write-up.md").read_text(encoding="utf-8")


def test_the_scripts_can_be_run():
    for script in ("start_local.sh", "stop_local.sh"):
        assert os.access(ROOT / "scripts" / script, os.X_OK), script


def test_the_website_pins_exact_versions():
    package = json.loads((ROOT / "web" / "package.json").read_text())
    versions = {**package["dependencies"], **package["devDependencies"]}
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", v) for v in versions.values()), versions
