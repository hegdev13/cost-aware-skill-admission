"""Build every figure from the CSVs in results/ (no numbers are hard-coded here).

    uv run python experiments/make_figures.py

Writes figs/<name>.pdf and figs/<name>.svg.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "figs"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from orx_figstyle import (BASELINE, COLUMN, CYCLE, MUTED, PALETTE, SEQUENTIAL, DIVERGING,
                          TEXT, WIDE, figure, figure_grid, panel_labels, save, use_style)

from ee_update import run_one
from ee_update.config import regime as mk_regime

RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

OURS = "c3_voi_propose"
OURS2 = "c1_voi"
CURVE_POLICIES = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote",
                  "c1_voi", "c3_voi_propose"]
SHOW = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote", "probe_gate",
        "always_verify", "c1_voi", "c3_voi_propose"]
LABEL = {
    "static": "never update",
    "accept_all": "accept all",
    "srn_like": "SRN-like",
    "greedy_judge": "greedy judge",
    "threshold_promote": "threshold promote",
    "probe_gate": "probe gate",
    "always_verify": "always verify",
    "c1_voi": "ours (no VoI gate)",
    "c3_voi_propose": "ours",
    "c3_greedy_mean": "ours, no information",
    "c3_probe_only": "ours, no verification",
    "c3_verify_only": "ours, no probation",
    "c3_no_judge": "ours, no judge",
    "c1_conservative": "ours, 1-sigma margin",
    "c1_risk_averse": "ours, 2-sigma margin",
    "c1_flat_prior": "ours, flat prior",
}
SHORT = {
    "static": "never\nupdate",
    "accept_all": "accept\nall",
    "srn_like": "SRN-like",
    "greedy_judge": "greedy\njudge",
    "threshold_promote": "thresh.\npromote",
    "probe_gate": "probe\ngate",
    "always_verify": "always\nverify",
    "c1_voi": "ours\n(no gate)",
    "c3_voi_propose": "ours",
}
SHORT_REG = {
    "base": "base", "A_cheap_verify": "cheap\nverify", "B_costly_verify": "costly\nverify",
    "C_costly_fail": "costly\nfailure", "D_weak_proposer": "weak\nproposer",
    "E_gen_gap": "general-\nisation gap", "F_drift": "drift", "G_dear_judge": "expensive\njudge",
    "H_noisy_judge": "noisy\njudge", "J_no_judge": "judge\nunavailable",
}
COLOR = {
    "static": BASELINE,
    "accept_all": PALETTE["orange"],
    "srn_like": PALETTE["purple"],
    "greedy_judge": PALETTE["green"],
    "threshold_promote": PALETTE["cyan"],
    "probe_gate": PALETTE["red"],
    "always_verify": "#999999",
    "c1_voi": PALETTE["yellow"],
    "c3_voi_propose": PALETTE["blue"],
}


def legend_handles(names):
    return [Line2D([], [], marker="o", color=COLOR[p], lw=0, markersize=4, label=SHORT[p])
            for p in names if p in COLOR]


# ------------------------------------------------------------------ loading
def load_grid():
    rows = list(csv.DictReader(open(RESULTS / "grid.csv")))
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k in ("regime", "policy", "seed"):
                continue
            d[(r["regime"], r["policy"])][k].append(float(v))
    return d, rows


def load_curves():
    rows = list(csv.DictReader(open(RESULTS / "curves.csv")))
    acc = defaultdict(list)
    for r in rows:
        acc[(r["regime"], r["policy"], int(r["seed"]))].append((int(r["t"]), float(r["regret"])))
    out = defaultdict(list)
    for (reg, pol, seed), pts in acc.items():
        pts.sort()
        out[(reg, pol)].append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    return out


def mean_ci(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)) * 1.96)


# ------------------------------------------------------------------ F1 curves
def fig_regret_curves(curves, regimes=("base", "J_no_judge")):
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.5, sharey=True)
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)
    for ax, reg in zip((ax0, ax1), regimes):
        for pol in CURVE_POLICIES:
            series = curves.get((reg, pol), [])
            if not series:
                continue
            ts = series[0][0]
            M = np.stack([s[1] for s in series if s[0].shape == ts.shape])
            mu = M.mean(axis=0)
            if len(M) > 1:
                half = M.std(axis=0, ddof=1) / np.sqrt(len(M)) * 1.96
                ax.fill_between(ts, mu - half, mu + half, color=COLOR[pol], alpha=0.18, linewidth=0)
            ax.plot(ts, mu, color=COLOR[pol], linewidth=1.5 if pol == OURS else 1.0,
                    label=SHORT[pol].replace("\n", " "), zorder=3 if pol == OURS else 2)
        ax.axhline(0, color=BASELINE, linewidth=0.7, linestyle=(0, (3, 3)))
        ax.set_xlabel("tasks seen")
        ax.set_xlim(0, None)
    ax0.set_ylabel("cumulative cost-regret (rollout-equivalents)")
    ax0.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
    ax1.text(0.03, 0.95, "judge unavailable", transform=ax1.transAxes, ha="left", va="top",
             fontsize=6.5, color=BASELINE,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    save(fig, str(FIGS / "fig1_regret_curves"))
    plt.close(fig)


# ------------------------------------------------------------------ F2 cost breakdown
def fig_cost_breakdown(d, regime="base"):
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.46, gridspec_kw={"width_ratios": [1.5, 1]})
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)
    pols = [p for p in SHOW if p != "always_verify"]
    comps = [("failures", "fail", PALETTE["red"]), ("task rollouts", "task_tokens", PALETTE["blue"]),
             ("proposals", "proposal_cost", PALETTE["orange"]),
             ("verification", "verify_cost", PALETTE["green"]),
             ("judge", "judge_cost", PALETTE["purple"])]
    cf = 5.0
    x = np.arange(len(pols))
    bottom = np.zeros(len(pols))
    for name, key, col in comps:
        vals = np.array([np.mean(d[(regime, p)]["failures"]) * cf if key == "fail"
                         else np.mean(d[(regime, p)][key]) for p in pols])
        ax0.bar(x, vals, bottom=bottom, color=col, width=0.74, label=name, linewidth=0)
        bottom += vals
    for xi, tot in zip(x, bottom):
        ax0.text(xi, tot, f"{tot:,.0f}", ha="center", va="bottom", fontsize=5.4, color="#333333")
    ax0.set_xticks(x)
    ax0.set_xticklabels([SHORT[p] for p in pols], fontsize=5.4)
    ax0.set_ylabel("total cost (rollout-equivalents)")
    ax0.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    ax0.set_ylim(0, bottom.max() * 1.2)

    vals = [np.mean(d[(regime, p)]["final_value"]) for p in pols]
    errs = [mean_ci(d[(regime, p)]["final_value"])[1] for p in pols]
    cols = [PALETTE["blue"] if p == OURS else MUTED for p in pols]
    ax1.bar(x, vals, yerr=errs, color=cols, width=0.74, linewidth=0, capsize=2,
            error_kw=dict(lw=0.6))
    ax1.axhline(np.mean(d[(regime, "static")]["final_value"]), color=BASELINE, lw=0.7,
                linestyle=(0, (3, 3)))
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT[p] for p in pols], fontsize=5.4)
    ax1.set_ylabel("final skill value (success probability)")
    save(fig, str(FIGS / "fig2_cost_breakdown"))
    plt.close(fig)


# ------------------------------------------------------------------ F3 pareto
def fig_pareto(d, regime="base"):
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.62)
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)
    for p in SHOW:
        if p == "always_verify":
            continue
        cost = np.mean(d[(regime, p)]["total_cost"])
        succ = np.mean(d[(regime, p)]["success_rate"])
        ce = mean_ci(d[(regime, p)]["total_cost"])[1]
        se = mean_ci(d[(regime, p)]["success_rate"])[1]
        ax0.errorbar(cost, succ, xerr=ce, yerr=se, fmt="o", color=COLOR[p], markersize=4,
                     capsize=2, lw=0.8, elinewidth=0.8, zorder=3 if p == OURS else 2)
    ax0.set_xlabel("total cost (rollout-equivalents)")
    ax0.set_ylabel("final success rate")
    ax0.legend(handles=legend_handles(SHOW), frameon=False, fontsize=5.8, loc="lower right")

    for p in SHOW:
        if p == "always_verify":
            continue
        over = (np.mean(d[(regime, p)]["proposal_cost"]) + np.mean(d[(regime, p)]["verify_cost"])
                + np.mean(d[(regime, p)]["judge_cost"]))
        succ = np.mean(d[(regime, p)]["success_rate"])
        ax1.errorbar(over, succ, fmt="o", color=COLOR[p], markersize=4, capsize=2, lw=0.8,
                     zorder=3 if p == OURS else 2)
    ax1.set_xlabel("update overhead (rollout-equivalents)")
    ax1.set_ylabel("final success rate")
    ax1.set_xscale("log")
    ax1.legend(handles=legend_handles(SHOW), frameon=False, fontsize=5.8, loc="upper left")
    save(fig, str(FIGS / "fig3_pareto"))
    plt.close(fig)


# ------------------------------------------------------------------ F4 regime map
def fig_regime_map(d, regimes):
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.5, gridspec_kw={"width_ratios": [1, 1.35]})
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)

    sw = list(csv.DictReader(open(RESULTS / "sweep.csv")))
    vps = sorted({float(r["verify_price"]) for r in sw})
    qs = sorted({float(r["q"]) for r in sw})
    fixed = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote", "probe_gate"]
    W = {}
    for r in sw:
        W.setdefault((qs.index(float(r["q"])), vps.index(float(r["verify_price"]))), {})[
            r["policy"]] = float(r["regret"])
    M = np.array([[W[(i, j)][OURS] - min(W[(i, j)][p] for p in fixed)
                  for j in range(len(vps))] for i in range(len(qs))])
    lim = max(abs(M).max(), 1)
    im = ax0.imshow(M, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax0.set_xticks(range(len(vps)))
    ax0.set_xticklabels([f"{v:g}" for v in vps])
    ax0.set_yticks(range(len(qs)))
    ax0.set_yticklabels([f"{q:g}" for q in qs])
    ax0.set_xlabel("verification price (rollout-equivalents per probe)")
    ax0.set_ylabel("proposer quality q")
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cb.set_label("regret of ours minus best fixed rule")
    cb.outline.set_linewidth(0.4)
    for i in range(len(qs)):
        for j in range(len(vps)):
            ax0.text(j, i, "ours" if M[i, j] < 0 else "fixed", ha="center", va="center",
                     fontsize=5.0, color="#111111")

    best_fixed = {}
    for rg in regimes:
        cands = [p for p in fixed if (rg, p) in d]
        best_fixed[rg] = min(cands, key=lambda p: np.mean(d[(rg, p)]["regret"]))
    for i, rg in enumerate(regimes):
        m1, e1 = mean_ci(d[(rg, OURS)]["regret"])
        m2, e2 = mean_ci(d[(rg, best_fixed[rg])]["regret"])
        ax1.errorbar(m2, i, xerr=e2, fmt="o", color=COLOR[best_fixed[rg]], markersize=4,
                     capsize=2, lw=0.8)
        ax1.errorbar(m1, i, xerr=e1, fmt="D", color=COLOR[OURS], markersize=4, capsize=2, lw=0.8)
    ax1.set_yticks(range(len(regimes)))
    ax1.set_yticklabels([SHORT_REG.get(r, r) for r in regimes], fontsize=5.8)
    ax1.set_xlabel("mean cost-regret (rollout-equivalents)")
    ax1.legend(handles=[Line2D([], [], marker="D", color=COLOR[OURS], lw=0, label="ours"),
                         Line2D([], [], marker="o", color=BASELINE, lw=0, label="best fixed rule")],
               frameon=False, fontsize=6, loc="upper left")
    ax1.set_ylim(len(regimes) - 0.4, -0.6)
    save(fig, str(FIGS / "fig4_regime_map"))
    plt.close(fig)


# ------------------------------------------------------------------ F6 ablations
def fig_ablations(d, regimes):
    use_style()
    variants = ["c3_voi_propose", "c1_voi", "c3_greedy_mean", "c3_probe_only", "c3_verify_only",
                "c3_no_judge", "c1_conservative", "c1_risk_averse", "c1_flat_prior"]
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.48, gridspec_kw={"width_ratios": [1, 1.25]})
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)

    rel = []
    for v in variants:
        ours = np.mean([np.mean(d[(r, OURS)]["regret"]) for r in regimes])
        rel.append(np.mean([np.mean(d[(r, v)]["regret"]) for r in regimes]) / ours)
    order = list(np.argsort(rel))
    y = np.arange(len(variants))
    ax0.barh(y, [rel[i] for i in order],
             color=[PALETTE["blue"] if variants[i] == OURS else MUTED for i in order],
             height=0.68, linewidth=0)
    ax0.axvline(1.0, color=BASELINE, lw=0.8, linestyle=(0, (3, 3)))
    ax0.set_yticks(y)
    ax0.set_yticklabels([LABEL[variants[i]] for i in order], fontsize=6.2)
    ax0.set_xlabel("mean regret relative to ours (>1 is worse)")
    ax0.set_xlim(0, max(rel) * 1.15)

    M = np.array([[np.mean(d[(r, v)]["regret"]) - np.mean(d[(r, OURS)]["regret"])
                   for r in regimes] for v in variants])
    im = ax1.imshow(M, cmap=DIVERGING, vmin=-np.abs(M).max(), vmax=np.abs(M).max(), aspect="auto")
    ax1.set_xticks(range(len(regimes)))
    ax1.set_xticklabels([SHORT_REG.get(r, r) for r in regimes], fontsize=4.8)
    ax1.set_yticks(range(len(variants)))
    ax1.set_yticklabels([LABEL[v] for v in variants], fontsize=6.2)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("regret minus ours")
    cb.outline.set_linewidth(0.4)
    save(fig, str(FIGS / "fig6_ablations"))
    plt.close(fig)


# ------------------------------------------------------------------ F5 decision timeline
def fig_decision_timeline(regime="base", seed=3, family=0):
    use_style()
    cfg = mk_regime(regime)
    r = run_one(cfg, OURS, seed, trace=True)
    tr = r["trace"]
    ev = tr.events
    truth = {t["skill_id"]: t for t in tr.truth if t.get("skill_id")}
    fam_events = [e for e in ev if e.get("family") == family]
    tasks = [e for e in fam_events if e["type"] == "TASK_SERVED"]
    actions = [e for e in fam_events if e["type"] == "ACTION_CHOSEN"]

    fig, axes = figure_grid(2, 1, width=WIDE, ratio=0.8)
    panel_labels(axes, ["(a)", "(b)"])
    ax0, ax1 = np.ravel(axes)

    ax0.plot([e["t"] for e in tasks], [0] * len(tasks), " ", label="tasks")
    ax0.scatter([e["t"] for e in tasks], [0] * len(tasks), s=9,
                c=["#0072B2" if e["outcome"] else "#D55E00" for e in tasks], marker="s",
                linewidth=0, zorder=3)
    marks = {"ACCEPT": ("v", PALETTE["blue"]), "SKIP": ("o", MUTED),
             "PROBATE": ("^", PALETTE["orange"]), "VERIFY": ("s", PALETTE["purple"]),
             "JUDGE": ("D", PALETTE["green"])}
    for e in actions:
        m, col = marks.get(e["action"], ("o", MUTED))
        hollow = e["mode"] == "EXPLORE"
        ax0.scatter([e["t"]], [0], marker=m, s=42, facecolor="none" if hollow else col,
                    edgecolors=col, linewidths=1.0, zorder=4)
    ax0.set_yticks([])
    ax0.set_ylim(-1, 1)
    ax0.set_xlabel("task index")
    ax0.set_ylabel(f"family {family}")
    ax0.legend(handles=[Line2D([], [], marker="s", color="#0072B2", lw=0, label="task success"),
                         Line2D([], [], marker="s", color="#D55E00", lw=0, label="task failure"),
                         Line2D([], [], marker="v", color=PALETTE["blue"], lw=0, label="accept"),
                         Line2D([], [], marker="^", markerfacecolor="none",
                                markeredgecolor=PALETTE["orange"], lw=0, label="probate (explore)"),
                         Line2D([], [], marker="D", markerfacecolor="none",
                                markeredgecolor=PALETTE["green"], lw=0, label="judge (explore)")],
                frameon=False, fontsize=6, ncol=5, loc="upper center")

    seq = [e for e in actions if (e.get("rule_inputs") or {}).get("mu_D") is not None]
    if seq:
        x = [e["t"] for e in seq]
        mu = np.array([e["rule_inputs"]["mu_D"] for e in seq])
        sd = np.array([e["rule_inputs"].get("s_D", 0.0) for e in seq])
        ax1.fill_between(x, mu - sd, mu + sd, color=PALETTE["blue"], alpha=0.18, linewidth=0)
        ax1.plot(x, mu, color=PALETTE["blue"], linewidth=1.3, label="belief mean")
        ax1.axhline(0, color=BASELINE, lw=0.7, linestyle=(0, (3, 3)))
        for e in seq:
            m, col = marks.get(e["action"], ("o", MUTED))
            hollow = e["mode"] == "EXPLORE"
            ax1.scatter([e["t"]], [e["rule_inputs"]["mu_D"]], marker=m, s=30,
                        facecolor="none" if hollow else col, edgecolors=col, linewidths=1.0,
                        zorder=4)
        ax1.set_xlabel("task index")
        ax1.set_ylabel("belief about v_cand - v_inc")
        ax1.legend(handles=[Line2D([], [], color=PALETTE["blue"], lw=1.3, label="belief mean +/- 1 sd"),
                             Line2D([], [], color=BASELINE, lw=0.7, linestyle=(0, (3, 3)),
                                    label="no difference")],
                   frameon=False, fontsize=6, loc="upper right")
    save(fig, str(FIGS / "fig5_decision_timeline"))
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    d, rows = load_grid()
    curves = load_curves()
    regimes = [r for r in dict.fromkeys(r["regime"] for r in rows)]
    fig_regret_curves(curves)
    fig_cost_breakdown(d)
    fig_pareto(d)
    fig_regime_map(d, regimes)
    fig_ablations(d, regimes)
    fig_decision_timeline()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
