# What the project will produce and how it will be shown

## 1. The deliverable in one paragraph
A short paper-style report with (a) a precise problem statement, (b) a decision rule for skill/memory updates that treats test cost as part of the objective, (c) a map showing in which cost regimes it beats simple baselines and where it does not, and (d) a small real-agent check. The core claim is conditional ("under these costs, this policy wins"), not "ours is best".

## 2. Core figures (each answers one question)

| # | Figure | Question it answers | What a good result looks like |
|---|---|---|---|
| F1 | Cumulative cost-regret vs tasks seen, one line per policy, shaded 95% CI over seeds | Does the controller learn faster and cheaper? | Our curve flattens earliest and lowest; accept-all grows linearly |
| F2 | Stacked bars of total cost per policy: task execution, proposal, probation, verification, wasted (never used) | Where does the money go? (mirrors RSI Tax) | Verification share collapses vs always-verify; waste share drops |
| F3 | Pareto plot: final success (y) vs total tokens (x) | Is it on the frontier? | Our points dominate or sit on the frontier; SRN is the cost anchor |
| F4 | Regime map: heatmap over verification-cost ratio × candidate-quality spread, colored by winning policy | When does smart updating matter? | Visible region where controller wins and region where SRN or accept-all wins |
| F5 | Decision timeline for one run: candidate arrivals, chosen action, posterior mean and band, ground truth marker | Is the behavior interpretable? | Reader can see it probe uncertain candidates and skip clearly bad ones |
| F6 | Ablation bars: no uncertainty, no VoI gate, no delayed credit, no cost term | Which piece matters? | Each removal hurts in at least one regime |
| F7 | Real-agent check: success vs tokens for 3 to 5 policies | Does the ordering carry over? | Same ordering as simulator, even if effect sizes differ |

## 3. Tables
- **T1 Main results:** per policy, mean ± CI for cost-regret, total tokens, final success, harmful commits, wasted-update share, across three named regimes (cheap verification, expensive verification, drifting tasks).
- **T2 Baseline definitions:** each baseline mapped to its source paper and the parameters we copied (e.g. m=2, ρ=0.67; probes 8/16/16).
- **T3 Simulator calibration:** every simulator parameter, its source number, and which paper it comes from.

## 4. Format and where files live
- **Report:** LaTeX paper draft (later), plus a short markdown summary for you after each phase.
- **Figures:** interactive HTML for exploration (F4 and F5 benefit most from hover) and static vector versions for the paper.
- **Raw data:** one CSV per experiment with seed, policy, regime, per-step costs; figures regenerate from CSV only.
- **Reproducibility:** each experiment config is a committed YAML with seed list; every figure has a script that rebuilds it from the CSVs.

## 5. Possible outcomes and what we would say
1. **Controller wins in many regimes:** headline = cost-aware E/E in admission reduces total cost by X% at equal success.
2. **Wins only in some regimes:** headline = the regime map; useful guidance on when to use which rule.
3. **Simple baseline always competitive:** headline = negative result extending RSI Tax: sophisticated update rules do not pay unless verification is cheap or candidates are high-variance.

All three are reportable; we decide the framing after P2, not before.
