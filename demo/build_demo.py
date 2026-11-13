"""Build the beginner-friendly interactive demo.

    uv run python demo/build_demo.py              # 50 seeds, matches the paper's numbers (~3-5 min)
    uv run python demo/build_demo.py --seeds 5    # quick preview
    open demo/demo.html

Writes one self-contained file, demo/demo.html (no server, no network).  It has three parts:

1. The race: every policy runs the same 1,000-task stream (same seeds, same candidates);
   watch the extra cost each one piles up, in six cost regimes.
2. Inside one decision: real decision records from the controller's log, explained in
   plain English, with the hidden truth revealed afterwards.
3. Be the controller: a JavaScript port of the controller's maths.  Move the sliders
   (prices, evidence) and see which action it picks and why.  The page checks its port
   against numbers computed here by the Python controller and shows the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ee_update import regime, run_one  # noqa: E402
from ee_update.policies import BeliefController, D_GRID  # noqa: E402
from ee_update.sim import ObsView, View  # noqa: E402

REGIMES = {
    "base": ("Base", "Default prices: a failure costs 5, a verification probe 1, a judge call 0.1."),
    "B_costly_verify": ("Verification is expensive", "Each held-out verification probe costs 2 instead of 1."),
    "C_costly_fail": ("Failures are very expensive", "A failed task costs 20 instead of 5. The one regime where ours loses."),
    "D_weak_proposer": ("Weaker proposer", "Good rewrites are half as likely (proposer quality 0.5)."),
    "G_dear_judge": ("Judge is pricier", "A pairwise judge call costs 1.0 instead of 0.1."),
    "J_no_judge": ("Judge is unaffordable", "A judge call costs 100. Rules that always ask the judge collapse."),
}
POLICIES = {
    "static": "Never update",
    "accept_all": "Accept everything",
    "srn_like": "Bounded note (SRN-style)",
    "threshold_promote": "Threshold promote (StepLearn-style)",
    "greedy_judge": "Greedy judge (EvoGuard-style)",
    "probe_gate": "Probe gate (FGSE-style)",
    "always_verify": "Always verify",
    "c3_voi_propose": "Ours (cost-aware controller)",
    "oracle": "Oracle (knows the truth)",
}
POINTS = 101


def _job(args):
    rname, pol, seed = args
    r = run_one(regime(rname), pol, seed)
    return rname, pol, seed, dict(
        cum=r["cum_cost"], total=r["total_cost"], fails=r["fails"], tasks=r["task_tokens"],
        proposal=r["proposal"], verify=r["verify"], judge=r["judge"], success=r["success_rate"],
        harmful=r["harmful_accepts"], accepts=r["accepts"], proposals=r["proposals"])


def race_data(seeds: int, procs: int) -> dict:
    jobs = [(rn, p, s) for rn in REGIMES for p in POLICIES for s in range(seeds)]
    # slow jobs first so the pool stays busy
    jobs.sort(key=lambda j: j[1] != "c3_voi_propose")
    res: dict = {}
    with Pool(procs) as pool:
        for i, (rn, p, s, d) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            res[(rn, p, s)] = d
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} runs", flush=True)
    out = {}
    idx = np.linspace(0, 999, POINTS).astype(int)
    for rn, (label, desc) in REGIMES.items():
        ref_curve = np.mean([np.minimum(res[(rn, "static", s)]["cum"], res[(rn, "oracle", s)]["cum"])
                             for s in range(seeds)], axis=0)
        ref_total = np.mean([min(res[(rn, "static", s)]["total"], res[(rn, "oracle", s)]["total"])
                             for s in range(seeds)])
        pols = {}
        for p in POLICIES:
            rows = [res[(rn, p, s)] for s in range(seeds)]
            curve = np.mean([r["cum"] for r in rows], axis=0) - ref_curve
            m = {k: float(np.mean([r[k] for r in rows])) for k in
                 ("total", "fails", "tasks", "proposal", "verify", "judge", "success", "harmful", "accepts", "proposals")}
            m["regret"] = m["total"] - float(ref_total)
            m["fail_cost"] = m["fails"] * regime(rn).c_fail
            m["curve"] = [round(float(x), 1) for x in curve[idx]]
            pols[p] = m
        out[rn] = dict(label=label, desc=desc, policies=pols)
    return out


def decision_stories(seed: int = 3) -> list:
    r = run_one(regime("base"), "c3_voi_propose", seed, trace=True)
    tr = r["trace"]
    truth = {t["skill_id"]: t for t in tr.truth if "true_value" in t}
    by_cand: dict = {}
    for e in tr.events:
        cid = e.get("candidate_id") or (e.get("skill_id") if e["type"] in ("SKILL_ACCEPTED", "SKILL_REJECTED") else None)
        if cid and "-c" in cid:
            by_cand.setdefault(cid, []).append(e)
    stories = {}
    for cid, evs in by_cand.items():
        acts = [e["action"] for e in evs if e["type"] == "ACTION_CHOSEN"]
        end = next((e["type"] for e in evs if e["type"] in ("SKILL_ACCEPTED", "SKILL_REJECTED")), None)
        if not acts or end is None:
            continue
        stories[cid] = dict(acts=acts, end=end, evs=evs)

    def pick(pred, n=1):
        got = [c for c, s in stories.items() if pred(s) and c not in chosen][:n]
        chosen.extend(got)

    chosen: list = []
    pick(lambda s: s["acts"][:1] == ["JUDGE"] and s["end"] == "SKILL_ACCEPTED" and len(s["acts"]) == 2)
    pick(lambda s: s["acts"][:1] == ["JUDGE"] and s["end"] == "SKILL_REJECTED" and len(s["acts"]) == 2)
    pick(lambda s: "VERIFY" in s["acts"])
    pick(lambda s: "PROBATE" in s["acts"])
    pick(lambda s: s["acts"].count("JUDGE") >= 2)
    # one mistake: accepted but truly not better
    for c, s in stories.items():
        if c not in chosen and s["end"] == "SKILL_ACCEPTED" and truth.get(c, {}).get("delta", 1) <= 0:
            chosen.append(c)
            break
    out = []
    for c in chosen:
        steps = []
        for e in stories[c]["evs"]:
            if e["type"] == "ACTION_CHOSEN":
                steps.append(dict(kind="decide", t=e["t"], action=e["action"], k=e.get("k", 0), mode=e["mode"],
                                  greedy=e.get("greedy_action"), inputs=e.get("rule_inputs", {}),
                                  scores=e.get("scores", {})))
            elif e["type"] == "JUDGE_RESULT":
                steps.append(dict(kind="judge", better=e["verdict_better"]))
            elif e["type"] == "VERIFY_RESULT":
                steps.append(dict(kind="verify", k=e["k"], ok=e["successes"]))
            elif e["type"] == "PROBATION_UPDATE":
                steps.append(dict(kind="probate", target=e.get("target_obs")))
            elif e["type"] in ("SKILL_ACCEPTED", "SKILL_REJECTED"):
                steps.append(dict(kind="end", accepted=e["type"] == "SKILL_ACCEPTED",
                                  n_live=e.get("n_live", 0), wins_live=e.get("wins_live", 0)))
        tv = truth.get(c, {})
        out.append(dict(id=c, family=int(c[1:c.index("-")]), steps=steps,
                        truth=dict(v=tv.get("true_value"), parent=tv.get("parent_true_value"), delta=tv.get("delta"))))
    return out


def port_check_cases() -> list:
    """Python controller scores for a set of inputs; the page recomputes them in JS."""
    rng = np.random.default_rng(7)
    cases = []
    for i in range(14):
        n = int(rng.integers(0, 25))
        vk = int(rng.choice([0, 0, 10, 20]))
        nj = int(rng.integers(0, 3))
        inc_w = int(rng.integers(10, 50))
        p = dict(c_fail=float(rng.choice([2, 5, 10, 20])), H=float(rng.integers(10, 250)),
                 judge_cost=float(rng.choice([0.05, 0.1, 1.0, 3.0])), verify_price=float(rng.choice([0.2, 1.0, 2.0])),
                 inc_n=60, inc_w=inc_w, n=n, w=int(rng.integers(0, n + 1)), vk=vk,
                 vs=int(rng.integers(0, vk + 1)), judge=[bool(x) for x in rng.integers(0, 2, nj)])
        pub = dict(c_fail=p["c_fail"], tau_prop=1.0, verify_price=p["verify_price"], judge_cost=p["judge_cost"],
                   judge_acc=0.7, q_p=0.2, opp_every=5)
        inc = ObsView("inc", 60, inc_w, 0, 0, (), 0, False)
        cand = ObsView("c", p["n"], p["w"], p["vk"], p["vs"], tuple(p["judge"]), 0, False)
        bc = BeliefController("c3_voi_propose", gate_propose=True)
        d = bc.decide(View(0, 0, inc, cand, p["H"], pub))
        cases.append(dict(inputs=p, scores={k: round(float(v), 6) for k, v in d.scores.items()}, action=d.action))
    return cases


def uses_binomial_predictive() -> bool:
    """True if the Python controller weights future counts by the binomial distribution.

    The page's JS port mirrors whichever version of `_trial_expected_gain` is in the repo.
    """
    bc = BeliefController()
    w = np.zeros_like(D_GRID)
    w[[60, 100]] = 0.5  # two point masses, D = -0.2 and +0.2
    im, n = 0.5, 2
    got = bc._trial_expected_gain(w, im, n)
    v = np.clip(im + D_GRID[[60, 100]], 1e-6, 1 - 1e-6)
    # binomial: accept iff the count makes the posterior mean positive (count >= 2 here)
    exact = 0.5 * 0.2 * v[1] ** 2 + 0.5 * -0.2 * v[0] ** 2
    return bool(abs(got - exact) < 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "demo" / "demo.html"))
    args = ap.parse_args()
    print(f"running {len(REGIMES)} regimes x {len(POLICIES)} policies x {args.seeds} seeds ...", flush=True)
    data = dict(seeds=args.seeds, regimes=race_data(args.seeds, args.procs), policyNames=POLICIES,
                stories=decision_stories(), portCheck=port_check_cases(),
                grid=[float(x) for x in D_GRID], binomialFix=uses_binomial_predictive())
    tpl = (ROOT / "demo" / "template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
