# ai-exploration-exploitation: Summary

## Project
Cost-aware explore/exploit controller for LLM agent skill admission. When a proposer offers a candidate rewrite of a skill, the agent must decide SKIP / ACCEPT / PROBATE / VERIFY / JUDGE, with every action priced in the same unit as task execution (1 task rollout = 1, a failed task costs an extra `c_fail`, a judge call costs `judge_cost`, a verification probe costs `verify_price`).

## Key Contribution
A cost-aware decision rule that computes the **expected net cost saving** of each action and picks the best. The value of information is computed **exactly** (enumerating every outcome on a grid posterior), not approximated. A live trial costs more than its failures because it blocks the family's candidate slot (opportunity cost).

## Verified Results

`experiments/verify_claims.py` re-derives 52 quoted numbers from the raw data (all pass); 54 tests pass.

### Simulator Study (P1–P2)
- 17 policies (7 baselines, 2 controllers, 7 ablations, and the free-information oracle) × 10 regimes × 50 seeds = 8,500 runs
- **Full controller: 610 ± 47 regret** vs greedy judge 752 ± 46 (± = 95% CI half-width); paired difference −142, 95% CI [−179, −105], better on 86% of seeds
- Lowest regret of every policy that does not see hidden values
- Final success 0.642, close to always-verify (0.647), using 7 verification probes instead of 19,776. The free-information oracle reaches 0.764.
- 20 harmful commits, against 100 for "accept every candidate" and 34 for the greedy judge

### Regime Performance (vs greedy judge, 95% CI)
| Regime | Ours − Judge | Outcome |
|---|---|---|
| base | −142 [−179, −105] | better |
| A_cheap_verify | −0 [−53, +53] | tie |
| B_costly_verify | −148 [−184, −113] | better |
| C_costly_fail | +115 [−121, +351] | tie |
| D_weak_proposer | −161 [−211, −111] | better |
| E_gen_gap | −143 [−180, −105] | better |
| F_drift | −251 [−333, −169] | better |
| G_dear_judge | −136 [−178, −95] | better |
| H_noisy_judge | −131 [−182, −79] | better |
| J_no_judge | −19,410 [−19,493, −19,326] | better (the judge baseline collapses when a comparison costs 100) |

**Against the greedy judge:** better in 8, tied in 2, worse in 0.
**Against the best fixed rule in each regime (the fairer count):** better in 6 (base, B, D, E, F, G), tied in 4 (A, C, H, J), worse in 0.

### Ablation (base regime; full controller = 610)
| Variant | Base Regret | Key Insight |
|---|---|---|
| Without propose gate | 677 | The gate is worth 67 units |
| Without verification channel | 671 | Paid probes add 61 units |
| Without probation channel | 683 | Live trials add 73 units |
| Without judge channel | 1,147 | Most valuable single channel (537 units) |
| Without any info channel | 1,177 | Buying info is worth 567 units |
| 1-sigma acceptance margin | 1,286 | Risk aversion destroys the rule (accepts nothing) |
| 2-sigma acceptance margin | 1,285 | As above |
| Flattened prior | 1,733 | Population prior carries much of the weight |

### Sweep (20 cells: verify_price × proposer_q, 20 seeds each)
- Controller wins **14 of 20 cells** (median margin −112 vs the best fixed rule in the cell)
- Weak proposer (q = 0.25): loses to never updating (962 vs 574); its prior assumes proposer quality 1.0

### Correction (2026-11-14)
The value-of-information enumeration previously dropped the binomial coefficient; it is now exact and tested. With the fix, the costly-failure regime changed from the one loss (+283) to a tie (+115 [−121, +351]); nothing else changed qualitatively.

## Real-LLM Replay (P3–P4): built and run
Recorded table in `data/replay/` (see `docs/06_replay_table_spec.md` §11 for deviations from the spec):
- 5 task families × 40 tasks with hidden conventions and exact-match verifiers; empty-skill success 40–42.5% per family
- 45 candidate skills written by the agent model (`gemini-3.5-flash-lite`) from its own real failures; 2,000 agent-task outcomes; 450 judge verdicts (`gemini-3.1-flash-lite`); real token counts
- Real world vs simulator: candidates 96% better / 4% tied / 0% worse (simulator 50/8/42%); mean gain +0.45 (simulator ~0.06); judge accuracy 0.76 (assumed 0.70); measured prices in task rollouts: judge 4.73 (assumed 0.1), proposal 6.27 (assumed 1), probe 1.19

| Policy | Regret, simulator prices | Regret, measured prices |
|---|---|---|
| oracle | 0 | 0 |
| **ours** | **54** | 281 |
| SRN-like bounded note | 91 | **85** |
| accept every candidate | 118 | 285 |
| greedy judge | 121 | 496 |
| never update | 1,075 | 1,005 |
| always verify | 4,578 | 5,579 |

- Simulator prices: ours − best fixed rule (SRN-like) = −37 [−62, −12]; ours − greedy judge = −67 [−87, −47]
- **Measured prices: ours loses to the SRN-like note, +196 [+150, +241]** (still beats the greedy judge by −216). Its fixed prior undervalues rewrites, so with proposals priced at 6.27 it stops after 4.5 proposals (SRN: 12.2) and reaches 0.793 task success vs 0.872.

## Key Design Properties
- **Grid posterior** over D = v_cand − v_inc, 181 points, prior calibrated to EvoGuard (42.2% worse, 7.7% tie, 50% better)
- **Exact VoI**: enumerate all binomial outcomes, recompute the posterior, count gain only if the candidate would be accepted
- **Opportunity cost**: a live trial blocks the family's candidate slot; every consumed task is a lost update opportunity
- **Propose gate**: ask the proposer only when the best action on a fresh candidate is worth more than the proposal
- **Seven baselines**: never update, accept every, SRN-like note, greedy judge, threshold promote, probe gate, always verify

## Limitations
- Simulated stream: costs in rollout-equivalents, not real tokens or dollars
- The simulator involves no language model; the real-LLM replay is small (one model family, 5 synthetic families, 45 candidates, 10 tasks per agent call)
- Judge accuracy fixed at 0.7; real comparisons are noisier and biased
- Skill value is a scalar per family; context-dependent skills not represented
- Exponential tail shape is an assumption; lognormal sensitivity not yet run

## Files
- `paper.tex` at repository root → compiles with `pdflatex paper.tex` (run twice for references)
- `data/replay/`: the recorded real-LLM table (committed)
- Simulator data is generated, not committed (`results/` is git-ignored). Regenerate and re-check with:
  ```sh
  uv run python experiments/run_grid.py --seeds 50   # results/grid.csv: 8,500 rows (~30 min)
  uv run python experiments/run_sweep.py --seeds 20  # results/sweep.csv
  uv run python experiments/run_replay.py            # results/replay_*.csv, replay_summary.json (~2 min, no API)
  uv run python experiments/verify_claims.py         # PASS/FAIL for every quoted number
  ```
- `docs/07_results_report.md`: results with figures and caveats
- `docs/04_simulator_spec.md`, `docs/06_replay_table_spec.md`: phase specifications
- Related work: EvoGuard 2611.06444, RSI Tax 2611.13524, FGSE 2611.14326, StepLearn 2609.35911, AtomSearch 2610.19951, SkillBoost 2607.26643, Skill∞ 2610.35434 (see `docs/01_literature_survey.md`)
