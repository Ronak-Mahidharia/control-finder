# ============================================================================
# Builds the answer key from NIST SP 800-171 Rev 3.
#
# Each of its 97 active requirements links to the SP 800-53 controls it was
# derived from (NIST's "source controls"). One requirement becomes one
# question: its statement is the question, and its source controls are the
# expected answers. The requirement's title is left out of the question,
# because it often repeats a control's title word for word.
#
# Questions are split into "dev" and "test" by a hash of the requirement
# number, so the split never changes. Settings are chosen on dev only.
#
# The second, harder key (load_csf_questions) comes from NIST's CSF 2.0: each of
# its 106 outcomes cites the SP 800-53 controls NIST maps it to. It's read from
# answer_keys/csf2_sp800_53.json, which scripts/build_csf_answer_key.py writes.
# ============================================================================
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from controlfinder.catalog import RAW, collect_params, flatten, prop, read_json, walk_controls

SP_800_171 = RAW / "NIST_SP800-171_rev3_catalog.json"
CSF_KEY = Path(__file__).resolve().parents[2] / "answer_keys" / "csf2_sp800_53.json"
SOURCE = re.compile(r"^([A-Z]{2})-(\d{2})(?:\((\d{2})\))?$")  # NIST's zero-padded labels, e.g. "AC-02(03)"


@dataclass(frozen=True)
class Question:
    id: str  # the requirement number, for example "03.01.01"
    title: str  # for example "Account Management"
    text: str  # the requirement's statement, with blanks written out
    expected: frozenset[str]  # OSCAL ids of its SP 800-53 source controls, for example {"ac-2", "ac-2.3"}
    split: str  # "dev" or "test"


def to_control_id(label):
    """'AC-02(03)' -> 'ac-2.3'; 'AC-02' -> 'ac-2'; anything else -> None."""
    m = SOURCE.match(label.strip())
    if not m:
        return None
    family, number, enhancement = m.groups()
    return f"{family.lower()}-{int(number)}" + (f".{int(enhancement)}" if enhancement else "")


def split_of(question_id):
    return "dev" if int(hashlib.sha256(question_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


@lru_cache(maxsize=1)
def load_questions():
    catalog = read_json(SP_800_171)["catalog"]
    params = collect_params(catalog)
    resources = {r["uuid"]: r for r in catalog.get("back-matter", {}).get("resources", [])}
    questions = []
    for group in catalog["groups"]:
        for requirement in walk_controls(group):
            if prop(requirement, "status") == "withdrawn":
                continue
            expected = set()
            for link in requirement.get("links", []) or []:
                resource = resources.get(link.get("href", "").lstrip("#"))
                if resource and any("SP_800_53" in r.get("href", "") for r in resource.get("rlinks", [])):
                    control_id = to_control_id(resource.get("title", ""))
                    if control_id:
                        expected.add(control_id)
            if not expected:
                continue
            statement = next(p for p in requirement["parts"] if p.get("name") == "statement")
            # Item labels like "SR-03.01.01.a" are left out; they'd only add noise to the question.
            text = " ".join(line.split(" ", 1)[1] if line.startswith("SR-") else line
                            for line in flatten(statement, params, "item"))
            number = prop(requirement, "sort-id") or requirement["id"]
            questions.append(Question(number, requirement["title"], text, frozenset(expected), split_of(number)))
    return tuple(questions)


@lru_cache(maxsize=1)
def load_csf_questions():
    """CSF 2.0 outcomes as questions; their mapped SP 800-53 controls are the right answers."""
    data = json.loads(CSF_KEY.read_text(encoding="utf-8"))
    return tuple(Question(q["id"], "", q["text"], frozenset(q["expected"]), split_of(q["id"])) for q in data["questions"])
