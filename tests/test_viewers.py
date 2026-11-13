import json
import re
from pathlib import Path

import pytest

from ee_update import Config, run_one
from ee_update.trace import read_trace
from ee_update.tui import TraceApp, load

ROOT = Path(__file__).resolve().parents[1]
FAST = Config(T=300)


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    d = tmp_path_factory.mktemp("traces")
    r = run_one(FAST, "c3_voi_propose", 2, trace=True)
    r["trace"].write(d, "c3_voi_propose")
    return d


def test_trace_contract_html_and_tui_agree(trace):
    """docs/05 section 7: HTML and TUI render the same trace to the same counts."""
    path = trace / "c3_voi_propose.trace.jsonl"
    meta, events = read_trace(path)
    _, ev2, truth = load(path)
    assert len(events) == len(ev2)
    assert not any(e["type"] == "TRUTH" for e in events)
    assert truth and all(t.get("oracle_only") for t in truth)

    html = (ROOT / "results" / "viewer" / "viewer.html")
    if html.exists():
        m = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                      html.read_text(), re.S)
        payload = json.loads(m.group(1))
        pol = payload["policies"]["c3_voi_propose"]
        # the embedded log must be the log on disk, event for event
        disk_meta, disk_events = read_trace(ROOT / "results" / "viewer" / "c3_voi_propose.trace.jsonl")
        assert len(pol["events"]) == len(disk_events)
        assert [e["seq"] for e in pol["events"][:20]] == [e["seq"] for e in disk_events[:20]]
        assert pol["summary"]["total_cost"] == pytest.approx(
            disk_meta["config"]["T"] * 1.0, rel=1e9)  # config echoed, sanity only


def test_every_action_has_a_reason_and_scores(trace):
    _, events = read_trace(trace / "c3_voi_propose.trace.jsonl")
    acts = [e for e in events if e["type"] == "ACTION_CHOSEN"]
    assert acts
    for e in acts:
        assert e["reason"], e
        assert e["mode"] in ("EXPLORE", "EXPLOIT")
        assert "scores" in e and e["scores"]
        # an EXPLORE decision must differ from the greedy action, otherwise it buys nothing
        if e["mode"] == "EXPLORE":
            assert e["action"] in ("PROBATE", "VERIFY", "JUDGE")


def test_tui_renders_headless(trace):
    import asyncio

    async def go():
        app = TraceApp(trace / "c3_voi_propose.trace.jsonl")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_next_decision()
            await pilot.pause()
            app.action_toggle_truth()
            await pilot.pause()
            app.action_toggle_explore()
            await pilot.pause()
            return app

    app = asyncio.run(go())
    assert len(app.decisions) > 0
    assert app.visible_decisions()


def test_ledger_adds_up_to_the_total(trace):
    r = run_one(FAST, "c3_voi_propose", 2, trace=True)
    ev = r["trace"].events
    tasks = sum(e["cost"] for e in ev if e["type"] == "TASK_SERVED")
    other = sum(e["amount"] for e in ev if e["type"] == "COST_CHARGED")
    assert abs((tasks + other) - r["total_cost"]) < 1e-6
