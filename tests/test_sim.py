import math
import numpy as np
import pytest

from ee_update import Config, regime, run_one, BASELINES, CONTROLLERS, ABLATIONS
from ee_update.sim import draw_delta, mixture_probs, get_world, View
from ee_update.trace import read_trace

ALL = BASELINES + CONTROLLERS + ABLATIONS + ["oracle"]
FAST = Config(T=400)


def test_determinism():
    a = run_one(FAST, "c3_voi_propose", 3)
    b = run_one(FAST, "c3_voi_propose", 3)
    assert a["total_cost"] == b["total_cost"]
    assert np.array_equal(a["cum_cost"], b["cum_cost"])


def test_calibration_matches_table():
    cfg = Config(diminishing=False)
    rng = np.random.default_rng(0)
    N = 400_000
    ut, um = rng.random(N), rng.random(N)
    d = np.array([draw_delta(cfg, 0.5, a, b) for a, b in zip(ut[:100_000], um[:100_000])])
    pw, pt, pb = mixture_probs(cfg)
    assert abs((d < 0).mean() - pw) < 0.01
    assert abs((d == 0).mean() - pt) < 0.01
    assert abs((d > 0).mean() - pb) < 0.01
    # EvoGuard tail masses: 7.7% lose >=0.10 and 11.3% gain >=0.10 (of all rewrites, renormalised by ~0.999)
    assert abs((d <= -0.10).mean() - 0.077) < 0.008
    assert abs((d >= 0.10).mean() - 0.113) < 0.008


@pytest.mark.parametrize("name", ALL)
def test_accounting_identity(name):
    r = run_one(FAST, name, 1, trace=True)
    ev = r["trace"].events
    task_cost = sum(e["cost"] for e in ev if e["type"] == "TASK_SERVED")
    other = sum(e["amount"] for e in ev if e["type"] == "COST_CHARGED")
    assert math.isclose(task_cost + other, r["total_cost"], rel_tol=1e-9)
    assert math.isclose(r["cum_cost"][-1], r["total_cost"], rel_tol=1e-9)


def test_oracle_dominates_when_proposals_free():
    cfg = Config(T=600, tau_prop=0.0)
    for s in range(3):
        o = run_one(cfg, "oracle", s)["total_cost"]
        for name in BASELINES + CONTROLLERS:
            assert o <= run_one(cfg, name, s)["total_cost"] + 1e-6, (name, s)


def test_static_beats_accept_all_without_good_candidates():
    cfg = Config(T=800, q=0.01)
    st = np.mean([run_one(cfg, "static", s)["total_cost"] for s in range(6)])
    aa = np.mean([run_one(cfg, "accept_all", s)["total_cost"] for s in range(6)])
    assert st < aa


def test_accept_all_beats_static_with_many_good_candidates():
    cfg = Config(T=800, q=3.0, tau_prop=0.2)
    st = np.mean([run_one(cfg, "static", s)["total_cost"] for s in range(6)])
    aa = np.mean([run_one(cfg, "accept_all", s)["total_cost"] for s in range(6)])
    assert aa < st


def test_views_hide_truth():
    # Views are frozen dataclasses with no hidden fields; policies only ever see them.
    fields = set(View.__dataclass_fields__)
    assert not any(("true" in f or "hidden" in f or "delta" in f) for f in fields)
    from ee_update.sim import ObsView
    assert not any(("true" in f or "hidden" in f or "delta" in f) for f in ObsView.__dataclass_fields__)
    for name in BASELINES + CONTROLLERS + ABLATIONS:
        from ee_update.policies import make_policy
        assert not make_policy(name).oracle


def test_crn_same_decisions_same_trajectory():
    # static and a controller that never proposes would match; check two accept_all runs with trace on/off
    a = run_one(FAST, "accept_all", 5, trace=False)
    b = run_one(FAST, "accept_all", 5, trace=True)
    assert np.array_equal(a["cum_cost"], b["cum_cost"])


def test_trace_contract(tmp_path):
    r = run_one(FAST, "c3_voi_propose", 2, trace=True)
    tr = r["trace"]
    tr.write(tmp_path, "x")
    meta, ev = read_trace(tmp_path / "x.trace.jsonl")
    created = {e["skill_id"] for e in ev if e["type"] == "SKILL_CREATED"}
    seed_skills = {f"f{f}-s0" for f in range(FAST.K)}
    served = {e["skill_id"] for e in ev if e["type"] == "TASK_SERVED"}
    assert served <= created | seed_skills
    # every accept/reject follows an ACTION_CHOSEN for the same skill
    chosen = set()
    for e in ev:
        if e["type"] == "ACTION_CHOSEN":
            chosen.add(e["candidate_id"])
        if e["type"] in ("SKILL_ACCEPTED", "SKILL_REJECTED"):
            assert e["skill_id"] in chosen
    # no TRUTH inside the policy-visible trace
    assert not any(e["type"] == "TRUTH" for e in ev)
    assert (tmp_path / "x.truth.jsonl").exists()


def test_trial_value_of_information_is_exact():
    """_trial_expected_gain must equal a brute-force enumeration with the binomial pmf."""
    from ee_update.policies import BeliefController, D_GRID
    from ee_update.sim import ObsView

    bc = BeliefController()
    # two-point belief: D = -0.2 or +0.2, incumbent 0.5, two new observations.
    # Accept only after 2/2 successes, so E[D*1{accept}] = 0.5*0.2*0.7^2 - 0.5*0.2*0.3^2 = 0.04
    w = np.zeros_like(D_GRID)
    w[[60, 100]] = 0.5
    assert abs(bc._trial_expected_gain(w, 0.5, 2) - 0.04) < 1e-9

    # a realistic posterior, against a brute-force loop
    w, _, _ = bc._belief(ObsView("c", 5, 3, 0, 0, (True,), 0, False), 0.55)
    for n, shift in [(3, 0.0), (16, 0.0), (20, 0.03)]:
        v = np.clip(0.55 + D_GRID + shift, 1e-6, 1 - 1e-6)
        want = 0.0
        for c in range(n + 1):
            pc = math.comb(n, c) * v ** c * (1 - v) ** (n - c)
            post = w * pc
            if (post / post.sum()) @ D_GRID > 0:
                want += float((w * D_GRID * pc).sum())
        assert abs(bc._trial_expected_gain(w, 0.55, n, shift) - want) < 1e-9
