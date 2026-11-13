"""Replay of recorded real-LLM outcomes: run on a small synthetic table, no API needed."""
import json
from pathlib import Path

import numpy as np
import pytest

from ee_update import BASELINES, CONTROLLERS, ABLATIONS
from ee_update.replay import ReplayTable, replay_config, run_replay
from ee_update.tasks import check, family_tasks, _strip_fences

ALL = BASELINES + CONTROLLERS + ABLATIONS + ["oracle"]
REAL = Path(__file__).resolve().parents[1] / "data" / "replay"


def _write_table(d: Path, families=2, cands=4, seed=0):
    rng = np.random.default_rng(seed)
    skills, cells, judge = [], [], []
    for f in range(families):
        ids = [f"f{f}-s0"] + [f"f{f}-c{k}" for k in range(cands)]
        for i, sid in enumerate(ids):
            p = 0.4 if i == 0 else float(rng.uniform(0.2, 0.8))
            outs = (rng.random(40) < p).astype(int)
            for t in range(40):
                cells.append(dict(skill_id=sid, task_id=f"f{f}-t{t:02d}", family=f, held_out=t >= 30,
                                  success=int(outs[t]), answer="", prompt_tokens=50.0, output_tokens=5.0,
                                  response_hash=""))
            skills.append(dict(id=sid, family=f, parent_id=None if i == 0 else ids[0], text="",
                               proposer_prompt_tokens=0 if i == 0 else 300, proposer_output_tokens=0 if i == 0 else 60,
                               true_value=float(outs.mean()), stream_value=float(outs[:30].mean()),
                               held_out_value=float(outs[30:].mean()), source_task_ids=[]))
        for a in ids:
            for b in ids:
                if a != b:
                    judge.append(dict(family=f, a=a, b=b, says="A" if rng.random() < 0.5 else "B",
                                      prompt_tokens=200, output_tokens=1))
    for name, rows in (("skills.jsonl", skills), ("cells.jsonl", cells), ("judge.jsonl", judge)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ReplayTable.load(d)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return _write_table(tmp_path_factory.mktemp("replay"))


CFG = replay_config(K=2, T=240)


@pytest.mark.parametrize("name", ALL)
def test_replay_accounting_identity(table, name):
    r = run_replay(table, CFG, name, 3)
    want = CFG.c_fail * r["fails"] + r["task_tokens"] + r["proposal"] + r["verify"] + r["judge"]
    assert abs(r["total_cost"] - want) < 1e-6
    assert r["token_cost"] > 0
    assert r["proposals"] <= 2 * 4      # never more candidates than the table holds


def test_replay_outcomes_come_from_the_table(table):
    """Never-update serves only the empty skills, so its failures are fixed by the table."""
    from ee_update.replay import ReplaySimulator
    from ee_update.policies import make_policy
    sim = ReplaySimulator(CFG, make_policy("static"), 5, table)
    r = sim.run()
    fails = 0
    seen = [0, 0]
    for t in range(CFG.T):
        f = int(sim.W.fam[t])
        task = int(sim.stream[f][seen[f] % 30])
        seen[f] += 1
        fails += 1 - table.cells[(table.roots[f], task)]["success"]
    assert r["fails"] == fails


def test_replay_is_deterministic(table):
    a = run_replay(table, CFG, "c3_voi_propose", 2)
    b = run_replay(table, CFG, "c3_voi_propose", 2)
    assert a["total_cost"] == b["total_cost"] and a["tokens"] == b["tokens"]


def test_replay_oracle_never_accepts_a_worse_skill(table):
    # every accepted candidate had a strictly higher true value than the incumbent it replaced
    tr = run_replay(table, CFG, "oracle", 1, trace=True)["trace"]
    for e in tr.truth:
        if e.get("event") == "accept":
            assert e["delta"] > 0


def test_verifier_strips_code_fences():
    t = family_tasks(3)[20]
    assert check(3, "```json\n" + t.expected + "\n```", t.expected)[0]
    assert _strip_fences("```\n42\n```") == "42"
    assert not check(0, "3/17/2024", "17/03/2024")[0]


def test_unit_conversion_prompts_match_their_answers():
    """Half the conversions run in reverse; the prompt must name the reverse direction."""
    for t in family_tasks(2)[16:]:
        value = float(t.prompt.split()[1])
        u1 = t.prompt.split()[2]
        exp = float(t.expected)
        if u1 == "celsius":
            assert abs(value * 9 / 5 + 32 - exp) <= 0.051
        if u1 == "fahrenheit":
            assert abs((value - 32) * 5 / 9 - exp) <= 0.051


@pytest.mark.skipif(not (REAL / "skills.jsonl").exists(), reason="replay table not built")
def test_recorded_table_is_complete():
    t = ReplayTable.load(REAL)
    meta = json.loads((REAL / "meta.json").read_text())
    n_skills = len(t.skills)
    assert len(t.cells) == n_skills * 40 == meta["agent_cells"]
    for f, cs in t.candidates.items():
        ids = [t.roots[f]] + cs
        assert all((a, b) in t.judge for a in ids for b in ids if a != b)
    for s in t.skills.values():
        got = np.mean([t.cells[(s["id"], i)]["success"] for i in range(40)])
        assert abs(got - s["true_value"]) < 1e-9
