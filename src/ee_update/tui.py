"""Terminal viewer for a decision trace (docs/05_decision_trace_and_ui_spec.md).

    uv run python -m ee_update.tui results/viewer/c3_voi_propose.trace.jsonl

Reads the same append-only log the HTML viewer renders, so the two cannot disagree.
Keys: space play/pause, n next decision, b previous, p switch policy, t toggle the
truth layer, e explore-only filter, / search a skill id, q quit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

from .trace import read_trace


def load(path: str | Path):
    path = Path(path)
    meta, events = read_trace(path)
    truth = []
    tpath = path.with_name(path.name.replace(".trace.jsonl", ".truth.jsonl"))
    if tpath.exists():
        for line in open(tpath):
            truth.append(json.loads(line))
    return meta, events, truth


class TraceApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 3; }
    #body { height: 1fr; }
    #stream { width: 46%; border: round $primary; }
    #side { width: 54%; border: round $primary; }
    .pane { padding: 0 1; }
    #belief { height: 12; border: round $primary; }
    #decision { height: 1fr; border: round $primary; }
    .muted { color: $text-muted; }
    """

    BINDINGS = [
        ("space", "toggle_play", "play/pause"),
        ("n", "next_decision", "next decision"),
        ("b", "prev_decision", "previous decision"),
        ("p", "switch_policy", "switch policy"),
        ("t", "toggle_truth", "truth layer"),
        ("e", "toggle_explore", "explore only"),
        ("/", "search", "search skill"),
        ("q", "quit", "quit"),
    ]

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.meta, self.events, self.truth = load(self.path)
        self.truth_map = {t["skill_id"]: t for t in self.truth if t.get("skill_id")}
        self.decisions = [e for e in self.events
                          if e["type"] in ("ACTION_CHOSEN", "PROPOSE_DECISION")]
        self.idx = 0
        self.show_truth = False
        self.explore_only = False
        self.playing = False
        self.siblings = sorted(p.parent for p in self.path.parent.glob("*.trace.jsonl"))

    # -------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="top")
        with Horizontal(id="body"):
            with Vertical(id="stream"):
                yield Static("task stream", classes="pane")
                yield DataTable(id="stream_table")
            with Vertical(id="side"):
                yield Static("belief about v_cand - v_inc", classes="pane")
                yield Static(id="belief")
                yield Static("decision log", classes="pane")
                yield Static(id="decision")
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#stream_table", DataTable)
        tbl.add_columns("t", "fam", "skill", "rule", "outcome", "cost")
        self.update_all()
        self.set_interval(0.35, self.tick)

    # -------------------------------------------------------------- helpers
    def visible_decisions(self):
        ds = self.decisions
        if self.explore_only:
            ds = [e for e in ds if e.get("mode") == "EXPLORE"]
        return ds

    def current(self):
        ds = self.visible_decisions()
        if not ds:
            return None
        return ds[min(self.idx, len(ds) - 1)]

    def update_all(self) -> None:
        self.render_top()
        self.render_stream()
        self.render_belief()
        self.render_decision()

    def render_top(self) -> None:
        ev = self.events
        tasks = [e for e in ev if e["type"] == "TASK_SERVED"]
        fails = sum(1 for e in tasks if not e["outcome"])
        cost = sum(e.get("cost", 0) for e in tasks)
        d = self.current()
        self.query_one("#top", Static).update(
            f"policy {self.meta.get('policy', '?')} | seed {self.meta.get('seed')} | "
            f"regime {self.meta.get('config', {}).get('regime', '-')} | "
            f"tasks {len(tasks)} | failures {fails} | task cost {cost:.0f} | "
            f"decision {self.idx + 1}/{len(self.visible_decisions())}"
            + (f" | t={d['t']} family {d['family']}" if d else "")
        )

    def render_stream(self) -> None:
        tbl = self.query_one("#stream_table", DataTable)
        tbl.clear()
        d = self.current()
        t_now = d["t"] if d else 10 ** 9
        rows = [e for e in self.events
                if e["type"] == "TASK_SERVED" and e["t"] <= t_now][-14:]
        for e in rows:
            tbl.add_row(str(e["t"]), str(e["family"]), e["skill_id"], e.get("routing_rule", ""),
                        "ok" if e["outcome"] else "FAIL", f"{e.get('cost', 0):.0f}")

    def render_belief(self) -> None:
        hist = [e for e in self.decisions
                if e["type"] == "ACTION_CHOSEN" and e.get("rule_inputs", {}).get("mu_D") is not None]
        hist = hist[-26:]
        if not hist:
            self.query_one("#belief", Static).update("no belief updates yet")
            return
        w = 52
        mus = [e["rule_inputs"]["mu_D"] for e in hist]
        sds = [e["rule_inputs"].get("s_D", 0.0) for e in hist]
        lo, hi = min(-0.05, min(m - s for m, s in zip(mus, sds))), max(0.05, max(m + s for m, s in zip(mus, sds)))
        lines = []
        for e, m, s in zip(reversed(hist), reversed(mus), reversed(sds)):
            pos = int((m - lo) / (hi - lo) * (w - 1))
            pos = max(0, min(w - 1, pos))
            bar = ["."] * w
            bar[pos] = "#"
            lo_p = int(((m - s) - lo) / (hi - lo) * (w - 1))
            hi_p = int(((m + s) - lo) / (hi - lo) * (w - 1))
            for i in range(max(0, lo_p), min(w, hi_p + 1)):
                if bar[i] == ".":
                    bar[i] = ":"
            mark = "*" if e["mode"] == "EXPLORE" else " "
            lines.append(f"{e['t']:>5} {e['action'][:8]:<9}{mark}|{''.join(bar)}| {m:+.3f}")
        self.query_one("#belief", Static).update("\n".join(lines[:11]))

    def render_decision(self) -> None:
        d = self.current()
        if d is None:
            self.query_one("#decision", Static).update("no decisions")
            return
        if d["type"] == "PROPOSE_DECISION":
            body = [f"PROPOSE_DECISION  chosen={d.get('chosen')}  mode={d.get('mode')}",
                    f"rule: {d.get('rule_id')}", "", d.get("reason", "")]
            scores = d.get("scores") or {}
        else:
            body = [f"{d['candidate_id']}  action={d['action']}"
                    + (f" (k={d['k']})" if d.get("k") else "")
                    + f"  mode={d['mode']}"
                    + (f"  greedy={d.get('greedy_action')}" if d.get("greedy_action") else ""),
                    f"rule: {d.get('rule_id')}", "", d.get("reason", "")]
            ri = d.get("rule_inputs") or {}
            if ri:
                body.append("")
                body.append("inputs:")
                for k, v in ri.items():
                    if k in ("probe_info", "verify_info", "judge_info"):
                        body.append(f"  {k}: {v}")
                    elif isinstance(v, float):
                        body.append(f"  {k}: {v:+.4f}")
                    else:
                        body.append(f"  {k}: {v}")
            scores = d.get("scores") or {}
            if scores:
                body.append("")
                body.append("alternatives (net value, cost units):")
                for k, v in scores.items():
                    body.append(f"  {k:<9}{v:+10.2f}" + ("   <- chosen" if k == d["action"] else ""))
        if self.show_truth:
            tr = self.truth_map.get(d.get("candidate_id"))
            body.append("")
            if tr:
                body.append(f"TRUTH: true value {tr.get('true_value')}"
                            + (f", delta vs parent {tr.get('delta'):+.4f}" if tr.get("delta") is not None else ""))
            else:
                body.append("TRUTH: (no truth entry for this skill)")
        self.query_one("#decision", Static).update("\n".join(body))

    # -------------------------------------------------------------- actions
    def action_toggle_play(self) -> None:
        self.playing = not self.playing

    def action_next_decision(self) -> None:
        self.idx = min(self.idx + 1, max(0, len(self.visible_decisions()) - 1))
        self.update_all()

    def action_prev_decision(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.update_all()

    def action_toggle_truth(self) -> None:
        self.show_truth = not self.show_truth
        self.update_all()

    def action_toggle_explore(self) -> None:
        self.explore_only = not self.explore_only
        self.idx = 0
        self.update_all()

    def action_switch_policy(self) -> None:
        if not self.siblings:
            return
        i = self.siblings.index(self.path)
        self.path = self.siblings[(i + 1) % len(self.siblings)]
        self.meta, self.events, self.truth = load(self.path)
        self.truth_map = {t["skill_id"]: t for t in self.truth if t.get("skill_id")}
        self.decisions = [e for e in self.events if e["type"] in ("ACTION_CHOSEN", "PROPOSE_DECISION")]
        self.idx = 0
        self.update_all()

    def action_search(self) -> None:
        from textual.widgets import Input
        if self.query_one("#searchbox", Input):
            return
        box = Input(placeholder="skill id to find (Enter to jump, Esc to cancel)")
        box.id = "searchbox"
        self.query_one("#body").mount(box)
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        needle = event.value.strip()
        self.remove_searchbox()
        if not needle:
            return
        matches = [i for i, e in enumerate(self.visible_decisions())
                   if needle in str(e.get("candidate_id", ""))]
        if matches:
            self.idx = matches[0]
        self.update_all()

    def on_input_blur(self, event: Input.Blurred) -> None:
        self.remove_searchbox()

    def remove_searchbox(self) -> None:
        try:
            self.query_one("#searchbox", Input).remove()
        except Exception:
            pass

    def tick(self) -> None:
        if not self.playing:
            return
        self.idx = min(self.idx + 1, max(0, len(self.visible_decisions()) - 1))
        if self.idx >= len(self.visible_decisions()) - 1:
            self.playing = False
        self.update_all()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m ee_update.tui <trace.jsonl>")
        raise SystemExit(2)
    TraceApp(sys.argv[1]).run()


if __name__ == "__main__":
    main()
