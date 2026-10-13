"""A record of every API request: which route, how long each step took, what the answer model used, and what went
wrong, so slow steps and failures show up without guessing.

- One record per request to /search, /answer, /controls, and /health, with an id the response sends back in
  X-Request-ID (the caller's own, if it sends a safe one).
- A question's text is never kept, only its length: the records can't leak what people asked.
- The last WINDOW records stay in memory for GET /stats (the median and 95th percentile of each step, and the error
  count); with CONTROLFINDER_TRACE_FILE set, each record is also added to that file as one line of JSON.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections import deque
from pathlib import Path

WINDOW = 1000
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def request_id(given: str | None) -> str:
    """The caller's request id if it's short and safe to echo, else a new one."""
    return given if given and SAFE_ID.fullmatch(given) else uuid.uuid4().hex[:16]


def percentile(values: list[float], share: float) -> float | None:
    """The nearest-rank percentile: the smallest value with at least `share` of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


class Tracer:
    def __init__(self, path: str | Path | None = None, window: int = WINDOW):
        self.path = Path(path) if path else None
        self.recent: deque[dict] = deque(maxlen=window)
        self.total = {"requests": 0, "errors": 0}
        self.lock = threading.Lock()

    def record(self, entry: dict) -> None:
        with self.lock:
            self.recent.append(entry)
            self.total["requests"] += 1
            self.total["errors"] += entry["status"] >= 500
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")

    def stats(self) -> dict:
        """Over the recent records: each route's count, errors, and total time, each step's time, and the answer
        model's tokens, as medians and 95th percentiles in milliseconds."""
        with self.lock:
            recent = list(self.recent)
            total = dict(self.total)
        routes, steps, tokens = {}, {}, {"prompt_tokens": [], "output_tokens": []}
        for e in recent:
            r = routes.setdefault(e["route"], {"requests": 0, "errors": 0, "ms": []})
            r["requests"] += 1
            r["errors"] += e["status"] >= 500
            r["ms"].append(e["ms"])
            for step, ms in e.get("steps", {}).items():
                steps.setdefault(step, []).append(ms)
            for key in tokens:
                if key in e:
                    tokens[key].append(e[key])

        def spread(values):
            return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "count": len(values)}
        return {"since_start": total, "window": len(recent),
                "routes": {k: {"requests": v["requests"], "errors": v["errors"], "ms": spread(v["ms"])}
                           for k, v in sorted(routes.items())},
                "steps_ms": {k: spread(v) for k, v in sorted(steps.items())},
                "answer_tokens": {k: spread(v) for k, v in tokens.items() if v}}
