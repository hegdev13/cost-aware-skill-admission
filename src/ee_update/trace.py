"""Append-only decision trace (docs/05_decision_trace_and_ui_spec.md).

One JSON object per event. A separate TRUTH layer holds hidden values and is only
written after the run, never shown to a policy.
"""
from __future__ import annotations

import json
from pathlib import Path


def _round(x, nd=4):
    if hasattr(x, "item") and not isinstance(x, (list, tuple, dict)):  # numpy scalar
        x = x.item()
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


class Trace:
    def __init__(self, enabled: bool = True, meta: dict | None = None):
        self.enabled = enabled
        self.events: list[dict] = []
        self.truth: list[dict] = []
        self.meta = meta or {}
        self._seq = 0

    def log(self, type_: str, t: int, family: int | None = None, **kw) -> int:
        if not self.enabled:
            return -1
        self._seq += 1
        ev = {"seq": self._seq, "type": type_, "t": t, "family": family}
        ev.update(_round(kw))
        self.events.append(ev)
        return self._seq

    def log_truth(self, **kw):
        if self.enabled:
            self.truth.append({"type": "TRUTH", "oracle_only": True, **_round(kw)})

    def write(self, directory: str | Path, name: str):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.trace.jsonl", "w") as f:
            f.write(json.dumps({"type": "META", **self.meta}) + "\n")
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        with open(d / f"{name}.truth.jsonl", "w") as f:
            for e in self.truth:
                f.write(json.dumps(e) + "\n")


def read_trace(path: str | Path):
    meta, events = {}, []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") == "META":
                meta = o
            else:
                events.append(o)
    return meta, events
