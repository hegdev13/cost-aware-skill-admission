"""Check every number quoted in docs/07_results_report.md and journal.md against the CSVs.

    uv run python experiments/verify_claims.py

Prints one line per claim with PASS or FAIL and exits non-zero if any claim fails.
This is the reproducible half of the verification log in journal.md section 9.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FAILS = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def load_grid():
    rows = list(csv.DictReader(open(RESULTS / "grid.csv")))
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k in ("regime", "policy", "seed"):
                continue
            d[(r["regime"], r["policy"])][k].append(float(v))
    return d, rows


def mean_ci(vals):
    v = list(map(float, vals))
    if len(v) < 2:
        return sum(v) / len(v), 0.0
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5 * 1.96


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    t = {5: 2.776, 10: 2.262, 20: 2.086, 30: 2.042, 50: 2.009}.get(n, 1.96)
    return m, t * sd / n ** 0.5, sum(1 for x in d if x < 0) / n


def main() -> None:
    d, rows = load_grid()
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    check("ten regimes in the grid", len(regimes) == 10, str(regimes))
    check("50 seeds per cell", all(len(d[(rg, "static")]["regret"]) == 50 for rg in regimes))

    def m(rg, pol, key):
        return mean_ci(d[(rg, pol)][key])[0]

    # ---- headline table in docs/07 (base regime)
    table = {
        "static": (1248, 3571, 514, 0, 0, 0, 0.486, 0),
        "accept_all": (1291, 3614, 483, 198, 0, 0, 0.517, 100),
        "srn_like": (1127, 3450, 472, 92, 0, 0, 0.528, 47),
        "threshold_promote": (1151, 3474, 490, 22, 0, 0, 0.510, 3),
        "greedy_judge": (752, 3075, 371, 198, 0, 20, 0.629, 34),
        "probe_gate": (5070, 7392, 448, 198, 3955, 0, 0.552, 2),
        "always_verify": (20417, 22740, 353, 198, 19776, 0, 0.647, 20),
        "c3_voi_propose": (623, 2946, 358, 115, 21, 21, 0.642, 20),
    }
    keys = ["regret", "total_cost", "failures", "proposals", "verify_cost", "judge_cost",
            "success_rate", "harmful_accepts"]
    for pol, want in table.items():
        got = tuple(round(m("base", pol, k)) for k in keys[:6]) + (
            round(m("base", pol, "success_rate"), 3), round(m("base", pol, "harmful_accepts")))
        ok = all(abs(g - w) <= max(1.0, abs(w) * 0.01) for g, w in zip(got, want))
        check(f"base-regime row for {pol}", ok, f"got {got} want {want}")

    # ---- paired comparison against the greedy judge, per regime
    expect_better = {"base", "B_costly_verify", "D_weak_proposer", "E_gen_gap", "F_drift",
                     "G_dear_judge", "H_noisy_judge", "J_no_judge"}
    expect_worse = {"C_costly_fail"}
    expect_tie = {"A_cheap_verify"}
    n_better = n_worse = n_tie = 0
    for rg in regimes:
        a = d[(rg, "c3_voi_propose")]["regret"]
        b = d[(rg, "greedy_judge")]["regret"]
        diff, ci, frac = paired(a, b)
        if rg in expect_better:
            ok = diff + ci < 0
            n_better += ok
            check(f"ours beats greedy judge in {rg}", ok, f"{diff:+.0f} [{diff - ci:+.0f}, {diff + ci:+.0f}]")
        elif rg in expect_worse:
            ok = diff - ci > 0
            n_worse += ok
            check(f"ours loses to greedy judge in {rg}", ok, f"{diff:+.0f} [{diff - ci:+.0f}, {diff + ci:+.0f}]")
        else:
            ok = diff - ci < 0 < diff + ci
            n_tie += ok
            check(f"ours ties greedy judge in {rg}", ok, f"{diff:+.0f} [{diff - ci:+.0f}, {diff + ci:+.0f}]")
    check("regime counts 8 better / 1 tie / 1 worse", (n_better, n_tie, n_worse) == (8, 1, 1),
          f"{n_better}/{n_tie}/{n_worse}")

    # ---- headline base-regime number quoted in the report
    a = d[("base", "c3_voi_propose")]["regret"]
    b = d[("base", "greedy_judge")]["regret"]
    diff, ci, frac = paired(a, b)
    check("base paired difference is -129 [-166, -91]",
          abs(diff + 129) < 1 and abs((diff - ci) + 166) < 2 and abs((diff + ci) + 91) < 2,
          f"{diff:+.1f} [{diff - ci:+.1f}, {diff + ci:+.1f}]")
    check("ours better on 86% of base seeds", abs(frac - 0.86) < 0.02, f"{frac:.0%}")

    # ---- ablation numbers quoted in docs/07 section 4
    abl = {"c1_voi": 694, "c3_probe_only": 671, "c3_verify_only": 700, "c3_no_judge": 1186,
           "c3_greedy_mean": 1177, "c1_conservative": 1515, "c1_risk_averse": 1315,
           "c1_flat_prior": 1711}
    for pol, want in abl.items():
        got = m("base", pol, "regret")
        check(f"ablation {pol} base regret", abs(got - want) <= max(2.0, want * 0.01),
              f"got {got:.0f} want {want}")
    check("the 1-sigma variant accepts nothing", m("base", "c1_conservative", "accepts") < 0.01,
          f"{m('base', 'c1_conservative', 'accepts'):.2f} accepts")

    # ---- sweep claims
    sw = list(csv.DictReader(open(RESULTS / "sweep.csv")))
    fixed = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote", "probe_gate"]
    cells = defaultdict(dict)
    for r in sw:
        cells[(float(r["verify_price"]), float(r["q"]))][r["policy"]] = float(r["regret"])
    wins, margins = 0, []
    for k, w in cells.items():
        mrg = w["c3_voi_propose"] - min(w[p] for p in fixed)
        margins.append(mrg)
        wins += int(mrg < 0)
    margins.sort()
    median = margins[len(margins) // 2]
    check("sweep: controller wins 13 of 20 cells", wins == 13, f"{wins}/{len(cells)}")
    check("sweep: median margin -102", abs(median + 102) < 2, f"{median:+.0f}")
    weak = cells[(0.5, 0.25)]
    check("sweep: weak proposer loses to never updating (965 vs 574)",
          abs(weak["c3_voi_propose"] - 965) < 12 and abs(weak["static"] - 574) < 12,
          f"{weak['c3_voi_propose']:.0f} vs {weak['static']:.0f}")

    # ---- the decision record quoted in docs/07 and journal.md
    trace = RESULTS / "viewer" / "c3_voi_propose.trace.jsonl"
    if trace.exists():
        evs = [json.loads(l) for l in open(trace)]
        rec = [e for e in evs if e.get("type") == "ACTION_CHOSEN" and e.get("candidate_id") == "f5-c9"]
        quoted = [e for e in rec if abs(e["rule_inputs"]["mu_D"] + 0.0161) < 1e-3]
        ok = bool(quoted) and quoted[0]["t"] == 304 and quoted[0]["action"] == "JUDGE" \
            and abs(quoted[0]["scores"]["JUDGE"] - 2.25) < 0.05 \
            and abs(quoted[0]["scores"]["PROBATE"] + 23.6) < 0.1
        check("quoted decision record for f5-c9 at t=304", ok,
              f"{len(rec)} decisions for f5-c9; the quoted one is seq {quoted[0]['seq'] if quoted else '-'}")
        check("the trace was built by the current controller (opportunity cost present)",
              any("v_opp" in ((e.get("rule_inputs") or {}).get("probe_info") or {}) for e in rec))
        check("no TRUTH inside the policy-visible log", not any(e.get("type") == "TRUTH" for e in evs))

    # ---- the metrics quoted in journal.md section 9
    meta = json.loads((RESULTS / "grid_meta.json").read_text())
    check("grid ran 50 seeds over 10 regimes", meta["seeds"] == 50 and len(meta["regimes"]) == 10)
    check("grid wall clock recorded", meta["seconds"] > 1000, f"{meta['seconds']:.0f}s")
    meta = json.loads((RESULTS / "sweep_meta.json").read_text())
    check("sweep ran 20 seeds over 20 cells", meta["seeds"] == 20 and len(meta["verify_prices"]) * len(meta["proposer_q"]) == 20)

    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} claims verified")
    if FAILS:
        print("FAILED: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
