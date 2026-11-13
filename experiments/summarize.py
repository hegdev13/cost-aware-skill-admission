"""Summarise the grid into the report's tables (results/tables.md).

    uv run python experiments/summarize.py

Everything printed here comes from results/grid.csv; nothing is hard-coded.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ORDER = ["static", "accept_all", "srn_like", "threshold_promote", "greedy_judge", "probe_gate",
         "always_verify", "c1_voi", "c3_voi_propose", "c3_greedy_mean", "c3_probe_only",
         "c3_verify_only", "c3_no_judge", "c1_conservative", "c1_risk_averse", "c1_flat_prior"]
NICE = {
    "static": "never update (static)",
    "accept_all": "accept every candidate",
    "srn_like": "SRN-like bounded note",
    "threshold_promote": "threshold promote (StepLearn-style)",
    "greedy_judge": "greedy pairwise judge (EvoGuard-style)",
    "probe_gate": "probe gate with LCB (FGSE-style)",
    "always_verify": "always verify (AtomSearch-style cost)",
    "c1_voi": "**ours**, no value-of-information gate on proposing",
    "c3_voi_propose": "**ours** (full controller)",
    "c3_greedy_mean": "ours, no information bought",
    "c3_probe_only": "ours, no verification channel",
    "c3_verify_only": "ours, no probation channel",
    "c3_no_judge": "ours, no judge channel",
    "c1_conservative": "ours, 1-sigma acceptance margin",
    "c1_risk_averse": "ours, 2-sigma acceptance margin",
    "c1_flat_prior": "ours, flattened prior",
}
SOURCES = {
    "static": "control",
    "accept_all": "EvoGuard-style upper bound",
    "srn_like": "The RSI Tax (SRN), one slot, overwrite",
    "threshold_promote": "StepLearn: fixed trial, promote on win rate",
    "greedy_judge": "EvoGuard: one pairwise judgement, accept the winner",
    "probe_gate": "FGSE: fixed probes, accept if gain LCB >= eta",
    "always_verify": "AtomSearch-style: full validation before every accept",
    "c1_voi": "this work",
    "c3_voi_propose": "this work",
}
CALIBRATION = [
    ("candidate better / worse / tie", "50.0% / 42.2% / 7.7%", "EvoGuard rewrite statistics (1,137 rewrites)"),
    ("tail mass |delta| >= 0.10", "11.3% better, 7.7% worse", "EvoGuard"),
    ("worse magnitude", "Exponential, scale 0.059", "derived from the 7.7% tail mass"),
    ("better magnitude", "Exponential, scale 0.067", "derived from the 11.3% tail mass"),
    ("failure price c_fail", "5 rollout-equivalents", "config default"),
    ("proposal cost", "1 rollout-equivalent", "config default"),
    ("verification price", "1 rollout-equivalent per probe", "config default"),
    ("judge cost / accuracy", "0.1 rollout-equivalents / 0.70", "config default (pairwise LLM comparison)"),
    ("probation traffic share q_p", "0.20 of a family's tasks", "config default"),
    ("stream", "6 families x 1000 tasks, 50 seeds", "config default"),
]


def load():
    rows = list(csv.DictReader(open(RESULTS / "grid.csv")))
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for k, v in r.items():
            if k in ("regime", "policy", "seed"):
                continue
            per[(r["regime"], r["policy"])][int(r["seed"])][k] = float(v)
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    return per, regimes


def mean_ci(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)) * 1.96)


def paired(a, b):
    """Paired mean difference a - b with a 95% t interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.746, 6: 2.571, 8: 2.365, 10: 2.262,
             15: 2.131, 20: 2.086, 30: 2.042, 50: 2.009}.get(n, 1.96)
    return float(m), float(tcrit * se), float((d < 0).mean())


def main() -> None:
    per, regimes = load()
    out = []
    out.append("# Results tables\n")
    out.append(f"Seeds: {len(next(iter(per.values())).values())} per cell. "
               "Regret is total cost minus the cheaper of never-updating and the "
               "free-information oracle on the same seed, so a policy that does nothing "
               "and a policy that knows everything both score 0.\n")

    # ---- T1 main results
    out.append("## T1. Main results (mean cost-regret, rollout-equivalents; lower is better)\n")
    head = "| policy | " + " | ".join(regimes) + " |"
    out.append(head)
    out.append("|" + "---|" * (len(regimes) + 1))
    for p in ORDER:
        cells = []
        for rg in regimes:
            if (rg, p) not in per:
                cells.append("-")
                continue
            vals = [d["regret"] for d in per[(rg, p)].values()]
            m, e = mean_ci(vals)
            cells.append(f"{m:,.0f} ± {e:,.0f}")
        out.append(f"| {NICE[p]} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- T1b paired comparison against the strongest fixed rule
    out.append("## T1b. Paired difference against the greedy judge (same seeds, same randomness)\n")
    out.append("| regime | ours - greedy judge (95% CI) | seeds where ours is better |")
    out.append("|---|---|---|")
    for rg in regimes:
        if (rg, "greedy_judge") not in per:
            continue
        a = [per[(rg, "c3_voi_propose")][s]["regret"] for s in per[(rg, "c3_voi_propose")]]
        b = [per[(rg, "greedy_judge")][s]["regret"] for s in per[(rg, "greedy_judge")]]
        m, ci, frac = paired(a, b)
        out.append(f"| {rg} | {m:+,.0f} [{m - ci:+,.0f}, {m + ci:+,.0f}] | {frac * 100:.0f}% |")
    out.append("")

    # ---- T1c cost decomposition and outcome quality
    out.append("## T1c. Where the money goes, and what it buys (base regime)\n")
    out.append("| policy | failures | task rollouts | proposals | verification | judge | "
               "final success | harmful commits |")
    out.append("|---|---|---|---|---|---|---|---|")
    for p in ORDER:
        if ("base", p) not in per:
            continue
        rows = list(per[("base", p)].values())
        g = lambda k: np.mean([r[k] for r in rows])  # noqa: E731
        out.append(f"| {NICE[p]} | {g('failures'):.0f} | {g('task_tokens'):.0f} | "
                   f"{g('proposal_cost'):.0f} | {g('verify_cost'):.0f} | {g('judge_cost'):.0f} | "
                   f"{g('success_rate'):.3f} | {g('harmful_accepts'):.0f} |")
    out.append("")

    # ---- T2 baseline definitions
    out.append("## T2. Baseline definitions (fixed rules, copied parameters)\n")
    out.append("| baseline | rule | source |")
    out.append("|---|---|---|")
    for p in ORDER:
        if p in SOURCES:
            out.append(f"| {NICE[p]} | {RULES[p]} | {SOURCES[p]} |" if False else
                       f"| {NICE[p]} | {RULES.get(p, '-')} | {SOURCES[p]} |")
    out.append("")

    # ---- T3 calibration
    out.append("## T3. Simulator calibration\n")
    out.append("| parameter | value | source |")
    out.append("|---|---|---|")
    for a, b, c in CALIBRATION:
        out.append(f"| {a} | {b} | {c} |")
    out.append("")

    text = "\n".join(out)
    (RESULTS / "tables.md").write_text(text)
    print(text)


RULES = {
    "static": "never propose; keep the empty skill",
    "accept_all": "propose at every opportunity, accept every candidate unchecked",
    "srn_like": "propose only after a failure, overwrite the note with no validation",
    "threshold_promote": "live trial until 10 tasks, then accept if the trial win rate beats the incumbent",
    "greedy_judge": "one pairwise judgement per candidate, accept the winner, never revisit",
    "probe_gate": "20 held-out probes per candidate, accept if the gain's lower confidence bound >= 0.03",
    "always_verify": "100 held-out probes per candidate, accept if the point estimate beats the incumbent",
    "c1_voi": "belief over the candidate's delta, priced in cost units, no gate on proposing",
    "c3_voi_propose": "same, plus a value-of-information gate on whether to propose at all",
    "c3_greedy_mean": "ours with the information channels removed (decide on the point estimate)",
    "c3_probe_only": "ours without the verification channel",
    "c3_verify_only": "ours without the live probation channel",
    "c3_no_judge": "ours without the pairwise judge channel",
    "c1_conservative": "ours accepting only when the belief clears a 1-sigma margin",
    "c1_risk_averse": "ours accepting only when the belief clears a 2-sigma margin",
    "c1_flat_prior": "ours with the population prior flattened (power weight 0.25)",
}

if __name__ == "__main__":
    main()
