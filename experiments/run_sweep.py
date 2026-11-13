"""Sweep: verification price x proposer quality, for the regime map figure.

Every cell runs the same policies on the same seeds, so the cell's winner and the
controller's margin over the best fixed rule are comparable across the grid.
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

from ee_update import Config, run_one

POLICIES = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote",
            "probe_gate", "c1_voi", "c3_voi_propose"]
VERIFY_PRICES = [0.2, 0.5, 1.0, 2.0, 4.0]
PROPOSER_Q = [0.25, 0.5, 1.0, 1.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for vp in VERIFY_PRICES:
        for q in PROPOSER_Q:
            cfg = Config(T=args.T, verify_price=vp, q=q)
            seeds = list(range(args.seeds))
            ref = {}
            for s in seeds:
                ref[s] = min(run_one(cfg, "static", s)["total_cost"],
                             run_one(cfg, "oracle", s)["total_cost"])
            for pol in POLICIES:
                reg = [run_one(cfg, pol, s)["total_cost"] - ref[s] for s in seeds]
                rows.append(dict(verify_price=vp, q=q, policy=pol,
                                 regret=float(np.mean(reg)),
                                 regret_sd=float(np.std(reg, ddof=1))))
            print(f"  verify={vp} q={q} done ({time.time() - t0:5.1f}s)", flush=True)

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "sweep_meta.json").write_text(json.dumps(
        dict(seeds=args.seeds, T=args.T, verify_prices=VERIFY_PRICES, proposer_q=PROPOSER_Q,
             policies=POLICIES, seconds=time.time() - t0), indent=2))
    print(f"wrote {out/'sweep.csv'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
