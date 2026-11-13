"""Replay every policy on the recorded real-LLM table (no API calls).

    uv run python experiments/run_replay.py            # 50 seeds, about a minute

Reads data/replay/ (built by experiments/build_replay_table.py) and writes
    results/replay.csv         one row per (policy, seed)
    results/replay_tables.md   the table statistics and the policy table
Seeds change the order of the stream tasks and of the candidates, not the outcomes:
every outcome is a recorded model answer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ee_update.policies import ABLATIONS, BASELINES, CONTROLLERS  # noqa: E402
from ee_update.replay import ReplayTable, replay_config, run_replay  # noqa: E402

POLICIES = BASELINES + CONTROLLERS + ABLATIONS + ["oracle"]
NAMES = {"static": "never update", "accept_all": "accept every candidate", "srn_like": "SRN-like bounded note",
         "threshold_promote": "threshold promote (StepLearn-style)", "greedy_judge": "greedy pairwise judge (EvoGuard-style)",
         "probe_gate": "probe gate with LCB (FGSE-style)", "always_verify": "always verify (AtomSearch-style cost)",
         "c3_voi_propose": "**ours**", "oracle": "free-information oracle"}


def table_stats(t: ReplayTable) -> dict:
    """Facts about the recorded table itself."""
    st = {}
    roots = {f: t.value(r) for f, r in t.roots.items()}
    deltas = [t.value(c) - roots[f] for f in t.candidates for c in t.candidates[f]]
    st["root_values"] = roots
    st["n_candidates"] = len(deltas)
    st["frac_better"] = float(np.mean([d > 0 for d in deltas]))
    st["frac_tie"] = float(np.mean([d == 0 for d in deltas]))
    st["frac_worse"] = float(np.mean([d < 0 for d in deltas]))
    st["mean_delta"] = float(np.mean(deltas))
    st["best_delta"] = float(max(deltas))
    # generalisation: stream vs held-out success of candidates
    gaps = [t.skills[c]["stream_value"] - t.skills[c]["held_out_value"] for f in t.candidates for c in t.candidates[f]]
    st["stream_minus_heldout"] = float(np.mean(gaps))
    # judge accuracy against the true values, over pairs with different values
    right = n = 0
    for (a, b), says in t.judge.items():
        va, vb = t.value(a), t.value(b)
        if va == vb or says not in "AB":
            continue
        n += 1
        right += (says == "A") == (va > vb)
    st["judge_pairs"] = n
    st["judge_accuracy"] = right / n if n else float("nan")
    st["judge_says_A"] = float(np.mean([s == "A" for s in t.judge.values()]))
    st["mean_task_tokens"] = t.mean_task_tokens()
    return st


def replay_one(table, cfg, seeds, out, label):
    """All policies x seeds at one price setting; returns (markdown lines, summary dict)."""
    rows = []
    ref = {}
    for s in range(seeds):
        costs = {p: run_replay(table, cfg, p, s) for p in ("static", "oracle")}
        ref[s] = (min(c["total_cost"] for c in costs.values()), min(c["token_cost"] for c in costs.values()))
    for p in POLICIES:
        for s in range(seeds):
            r = run_replay(table, cfg, p, s)
            rows.append(dict(policy=p, seed=s, total_cost=r["total_cost"], regret=r["total_cost"] - ref[s][0],
                             token_cost=r["token_cost"], token_regret=r["token_cost"] - ref[s][1],
                             success_rate=r["success_rate"], final_value=r["final_value"], failures=r["fails"],
                             proposals=r["proposals"], verify_cost=r["verify"], judge_cost=r["judge"],
                             accepts=r["accepts"], harmful_accepts=r["harmful_accepts"],
                             **{f"tokens_{k}": v for k, v in r["tokens"].items()}))
        print(f"  {p} done", flush=True)
    with open(out / f"replay_{label}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k not in ("policy", "seed"):
                by[r["policy"]][k].append(float(v))

    def m(p, k):
        return float(np.mean(by[p][k]))

    def paired(a, b):
        d = np.array(by[a]["regret"]) - np.array(by[b]["regret"])
        hw = 2.01 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        return float(d.mean()), float(hw), float(np.mean(d < 0))

    pr = dict(tau_prop=cfg.tau_prop, verify_price=cfg.verify_price, judge_cost=cfg.judge_cost)
    lines = [f"## Prices: {label} (proposal {pr['tau_prop']:.2f}, probe {pr['verify_price']:.2f}, "
             f"judge {pr['judge_cost']:.2f}; failure {cfg.c_fail:.0f}; one task rollout = 1)", "",
             "| policy | regret | total cost | token regret | failures | proposals | verification | judge | success | harmful commits |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(POLICIES, key=lambda p: m(p, "regret"))
    for p in order:
        lines.append(f"| {NAMES.get(p, p)} | {m(p, 'regret'):,.0f} | {m(p, 'total_cost'):,.0f} | "
                     f"{m(p, 'token_regret'):,.0f} | {m(p, 'failures'):.0f} | {m(p, 'proposals'):.1f} | "
                     f"{m(p, 'verify_cost'):,.0f} | {m(p, 'judge_cost'):.1f} | {m(p, 'success_rate'):.3f} | "
                     f"{m(p, 'harmful_accepts'):.1f} |")
    fixed = [p for p in BASELINES]
    best = min(fixed, key=lambda p: m(p, "regret"))
    d, hw, frac = paired("c3_voi_propose", best)
    dj, hwj, fracj = paired("c3_voi_propose", "greedy_judge")
    lines += ["", f"Ours minus the best fixed rule ({best}): {d:+.0f} [{d - hw:+.0f}, {d + hw:+.0f}], "
                  f"ours better on {frac:.0%} of seeds.",
              f"Ours minus the greedy judge: {dj:+.0f} [{dj - hwj:+.0f}, {dj + hwj:+.0f}], better on {fracj:.0%} of seeds.", ""]
    summary = dict(best_fixed=best, ours_minus_best=[d, hw, frac], ours_minus_judge=[dj, hwj, fracj], prices=pr,
                   means={p: {k: m(p, k) for k in ("regret", "total_cost", "token_regret", "token_cost",
                                                   "success_rate", "failures", "proposals", "verify_cost",
                                                   "judge_cost", "harmful_accepts", "accepts")} for p in POLICIES})
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "data" / "replay"))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    table = ReplayTable.load(args.table)
    cfg = replay_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = table_stats(table)
    prices = {"assumed": {}, "measured": table.measured_prices()}
    lines, summary = [], dict(stats=st, seeds=args.seeds, prices=prices)
    for label, pr in prices.items():
        cfg = replay_config(**pr)
        l, sm = replay_one(table, cfg, args.seeds, out, label)
        lines += l
        summary[label] = sm
    head = ["# Real-LLM replay results", "",
            f"Table: {table.families} families, {st['n_candidates']} proposer candidates, "
            f"{len(table.cells)} recorded agent-task outcomes, {len(table.judge)} judge verdicts. "
            f"Replay: {args.seeds} seeds x {replay_config().T} tasks.", "",
            "## The table", "",
            f"- empty-skill true value per family: " + ", ".join(f"{v:.3f}" for v in st["root_values"].values()),
            f"- candidates vs the empty skill: {st['frac_better']:.0%} better, {st['frac_tie']:.0%} tied, "
            f"{st['frac_worse']:.0%} worse; mean change {st['mean_delta']:+.3f}, best {st['best_delta']:+.3f}",
            f"- stream minus held-out success of candidates: {st['stream_minus_heldout']:+.3f}",
            f"- pairwise judge accuracy on {st['judge_pairs']} pairs with different true values: "
            f"{st['judge_accuracy']:.3f} (says 'A' {st['judge_says_A']:.0%} of the time)",
            f"- mean tokens per agent-task: {st['mean_task_tokens']:.1f}",
            "- measured prices in task-rollout units: " + ", ".join(f"{k} {v:.2f}" for k, v in prices["measured"].items()),
            ""]
    text = "\n".join(head + lines) + "\n"
    (out / "replay_tables.md").write_text(text)
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    print(text)


if __name__ == "__main__":
    main()
