# ============================================================================
# Reads NIST SP 800-53 Rev 5.2.0, with the SP 800-53A Rev 5.2.0 assessment
# objectives, from NIST's OSCAL JSON file.
#
# Each control and enhancement becomes one Control record: its label (IA-5(1)),
# title, family, statement, guidance, and assessment objectives as plain text.
# Fill-in blanks are written the way NIST prints them, for example
# "[Assignment: organization-defined frequency]" or "[Selection (one or more): ...]".
# Run scripts/download_data.py first.
# ============================================================================
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
SP_800_53 = RAW / "NIST_SP-800-53_rev5_catalog.json"
INSERT = re.compile(r"\{\{\s*insert:\s*param,\s*([^}\s]+)\s*\}\}")
ID = re.compile(r"^([a-z]{2})-(\d{1,2})(?:\((\d{1,2})\)|\.(\d{1,2}))?$", re.I)


def to_id(text: str) -> str | None:
    """'IA-5(1)', 'ia-5.1', and 'IA-05(01)' all become 'ia-5.1'."""
    m = ID.match(text.strip())
    if not m:
        return None
    family, number, enhancement = m.group(1).lower(), int(m.group(2)), m.group(3) or m.group(4)
    return f"{family}-{number}" + (f".{int(enhancement)}" if enhancement else "")


@dataclass(frozen=True)
class Control:
    id: str  # OSCAL id, for example "ia-5.1"
    label: str  # as NIST prints it, for example "IA-5(1)"
    title: str  # for example "Password-based Authentication"
    family: str  # for example "Identification and Authentication"
    parent: str | None  # "ia-5" for an enhancement, None for a base control
    statement: str
    guidance: str
    objectives: tuple[str, ...]  # SP 800-53A assessment objectives that have text
    objective_parts: int  # every assessment-objective part, including the ones that only group others
    withdrawn: bool


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run: python scripts/download_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


def prop(node, name, cls=None):
    """The value of the first property with this name (and class, if given)."""
    for p in node.get("props", []) or []:
        if p.get("name") == name and p.get("class") == cls:
            return p.get("value")
    return None


def walk_controls(node):
    for control in node.get("controls", []) or []:
        yield control
        yield from walk_controls(control)


def collect_params(catalog):
    params = {}
    for group in catalog["groups"]:
        for control in walk_controls(group):
            for param in control.get("params", []) or []:
                params[param["id"]] = param
    return params


def render(text, params):
    """Replace {{ insert: param, ... }} with the blank as NIST prints it."""
    return INSERT.sub(lambda m: render_param(params.get(m.group(1)), params), text or "")


def render_param(param, params):
    if param is None:
        return "[Assignment]"
    if "select" in param:
        select = param["select"]
        choices = "; ".join(render(choice, params) for choice in select.get("choice", []))
        head = "Selection (one or more)" if select.get("how-many") == "one-or-more" else "Selection"
        return f"[{head}: {choices}]"
    label = param.get("usage") or param.get("label") or "value"
    if not label.startswith("organization-defined"):
        label = f"organization-defined {label}"
    return f"[Assignment: {label}]"


def flatten(part, params, child_name):
    """A part's text, then its children's, each led by its label (a., 1., (a))."""
    lines = []
    label = prop(part, "label") or ""
    prose = render(part.get("prose", ""), params)
    if prose:
        lines.append(f"{label} {prose}".strip())
    for child in part.get("parts", []) or []:
        if child.get("name") == child_name:
            lines.extend(flatten(child, params, child_name))
    return lines


def objective_texts(control, params):
    texts, count = [], 0

    def visit(part):
        nonlocal count
        count += 1
        if part.get("prose"):
            texts.append(render(part["prose"], params))
        for child in part.get("parts", []) or []:
            if child.get("name") == "assessment-objective":
                visit(child)

    for part in control.get("parts", []) or []:
        if part.get("name") == "assessment-objective":
            visit(part)
    return tuple(texts), count


@lru_cache(maxsize=1)
def load_catalog():
    """Every control and enhancement in the catalog, withdrawn ones included."""
    catalog = read_json(SP_800_53)["catalog"]
    params = collect_params(catalog)
    controls = []
    for group in catalog["groups"]:
        def add(node, parent):
            parts = node.get("parts", []) or []
            statement = next((p for p in parts if p.get("name") == "statement"), None)
            guidance = next((p for p in parts if p.get("name") == "guidance"), None)
            objectives, objective_parts = objective_texts(node, params)
            controls.append(Control(
                id=node["id"],
                label=prop(node, "label") or node["id"].upper(),
                title=node["title"],
                family=group["title"],
                parent=parent,
                statement=" ".join(flatten(statement, params, "item")) if statement else "",
                guidance=render(guidance.get("prose", ""), params) if guidance else "",
                objectives=objectives,
                objective_parts=objective_parts,
                withdrawn=prop(node, "status") == "withdrawn",
            ))
            for child in node.get("controls", []) or []:
                add(child, node["id"])

        for control in group.get("controls", []) or []:
            add(control, None)
    return tuple(controls)


def active_controls():
    """The controls and enhancements a search can return: everything not withdrawn."""
    return tuple(c for c in load_catalog() if not c.withdrawn)


def families():
    return tuple(dict.fromkeys(c.family for c in load_catalog()))
