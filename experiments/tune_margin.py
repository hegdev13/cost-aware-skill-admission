"""The hyperparameter check that fixed the controller's acceptance margin.

    uv run python experiments/tune_margin.py

Sweeps the acceptance margin z and the gate on proposing over the four *development*
regimes (base, D_weak_proposer, F_drift, G_dear_judge).  The regimes not used here
(A, B, C, E, H, J) were left untouched by any choice.

Writes results/tuning.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ee_update import run_one
from ee_update.config import regime as mk_regime
from ee_update.policies import BeliefController
from ee_update.sim import Simulator

DEV_REGIMES = ["base", "D_weak_proposer", "F_drift", "G_dear_judge"]
MARGINS = [0.0, 0.15, 0.3, 0.5]
SEEDS = 25


def regret(policy, cfg, seeds=SEEDS) -> float:
    out = []
    for s in range(seeds):
        ref = min(run_one(cfg, "static", s)["total_cost"], run_one(cfg, "oracle", s)["total_cost"])
        out.append(Simulator(cfg, policy, s).run()["total_cost"] - ref)
    return float(np.mean(out))


def main() -> None:
    t0 = time.time()
    rows = []
    for z in MARGINS:
        for gate in (True, False):
            vals = []
            for rg in DEV_REGIMES:
                cfg = mk_regime(rg)
                pol = BeliefController(f"z{z}_gate{int(gate)}", z=z, gate_propose=gate)
                vals.append(regret(pol, cfg))
            rows.append(dict(margin=z, gate_propose=int(gate),
                             **{rg: v for rg, v in zip(DEV_REGIMES, vals)},
                             mean=float(np.mean(vals))))
            print(f"  z={z} gate={int(gate)}  " +
                  "  ".join(f"{rg[:6]}={v:7.0f}" for rg, v in zip(DEV_REGIMES, vals)) +
                  f"  mean={np.mean(vals):7.0f}   ({time.time() - t0:5.1f}s)", flush=True)
    out = ROOT / "results" / "tuning.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["mean"])
    print(f"\nbest: margin {best['margin']}, gate {bool(best['gate_propose'])}, "
          f"mean regret {best['mean']:.0f}   (written to {out})")


if __name__ == "__main__":
    main()
