"""Experiment grid: policies x regimes x seeds.

Writes one CSV of per-run metrics (results/grid.csv) and one CSV of the cumulative
cost-regret curves (results/curves.csv).  Every run of every policy sees the same
pre-drawn randomness for a given seed (common random numbers), so paired differences
between policies are low-variance.

Usage:
    uv run python experiments/run_grid.py --seeds 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ee_update import Config, REGIMES, regime, run_one
from ee_update.policies import BASELINES, CONTROLLERS, ABLATIONS

ALL_POLICIES = BASELINES + CONTROLLERS + ABLATIONS + ["oracle"]
REF_POLICIES = ["static", "oracle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--regimes", default="all")
    ap.add_argument("--policies", default="all")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    regimes = list(REGIMES) if args.regimes == "all" else args.regimes.split(",")
    policies = ALL_POLICIES if args.policies == "all" else args.policies.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows, curve_rows = [], []
    for rname in regimes:
        cfg = regime(rname, T=args.T)
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        # reference per seed: the cheaper of never-updating and the free-information oracle
        ref, ref_curve = {}, {}
        for s in seeds:
            costs, curves = {}, {}
            for p in REF_POLICIES:
                r = run_one(cfg, p, s)
                costs[p] = r["total_cost"]
                curves[p] = r["cum_cost"]
            ref[s] = min(costs.values())
            ref_curve[s] = np.minimum(curves["oracle"], curves["static"])
        for pol in policies:
            for s in seeds:
                r = run_one(cfg, pol, s)
                rows.append(dict(
                    regime=rname, policy=pol, seed=s,
                    total_cost=r["total_cost"], regret=r["total_cost"] - ref[s],
                    success_rate=r["success_rate"], final_value=r["final_value"],
                    failures=r["fails"], task_tokens=r["task_tokens"],
                    proposal_cost=r["proposal"], verify_cost=r["verify"], judge_cost=r["judge"],
                    proposals=r["proposals"], accepts=r["accepts"],
                    harmful_accepts=r["harmful_accepts"], rejects=r["rejects"],
                    rejected_good=r["rejected_good"], explore_actions=r["explore_actions"],
                    rejected_tokens=r["rejected_tokens"],
                ))
                cum = r["cum_cost"]
                rc = ref_curve[s]
                step = max(1, len(cum) // 200)
                for t in range(0, len(cum), step):
                    curve_rows.append(dict(regime=rname, policy=pol, seed=s, t=t,
                                           cum_cost=float(cum[t]),
                                           regret=float(cum[t] - min(rc[t], cum[t]))))
            print(f"  {rname:16s} {pol:20s} done ({time.time() - t0:6.1f}s)", flush=True)

    keys = list(rows[0].keys())
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)
    meta = dict(seeds=args.seeds, seed0=args.seed0, T=args.T, regimes=regimes,
                policies=policies, seconds=time.time() - t0)
    (out / "grid_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'grid.csv'} ({len(rows)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
