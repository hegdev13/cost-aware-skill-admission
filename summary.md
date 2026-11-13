# ai-exploration-exploitation: Summary

## Project
Cost-aware explore/exploit controller for LLM agent skill admission. When a proposer offers a candidate rewrite of a skill, the agent must decide SKIP / ACCEPT / PROBATE / VERIFY / JUDGE, with every action priced in the same unit as task execution (1 task rollout = 1, a failed task costs an extra `c_fail`, a judge call costs `judge_cost`, a verification probe costs `verify_price`).

## Key Contribution
A cost-aware decision rule that computes the **expected net cost saving** of each action and picks the best. The value of information is computed **exactly** (enumerating every outcome on a grid posterior), not approximated. A live trial costs more than its failures because it blocks the family's candidate slot (opportunity cost).

## Verified Results (simulator only)

### Simulator Study (P1–P2)
- 30/30 pytest tests pass; `experiments/verify_claims.py` re-derives 41 quoted numbers from the raw CSVs
- 17 policies (7 baselines, 2 controllers, 7 ablations, and the free-information oracle) × 10 regimes × 50 seeds = 8,500 runs
- **Full controller: 623 ± 47 regret** vs greedy judge 752 ± 46 (± = 95% CI half-width); paired difference −129, 95% CI [−166, −91], better on 86% of seeds
- Lowest regret of every policy that does not see hidden values
- Final success 0.642, close to always-verify (0.647), using 21 verification probes instead of 19,776 (about 1/940 of the verification cost). The free-information oracle reaches 0.764.
- 20 harmful commits, against 100 for "accept every candidate" and 34 for the greedy judge ("always verify" also makes 20, at ~20,000 rollout-equivalents of verification)

### Regime Performance (vs greedy judge, 95% CI)
| Regime | Ours − Judge | Outcome |
|---|---|---|
| base | −129 [−166, −91] | better |
| A_cheap_verify | +18 [−38, +74] | tie |
| B_costly_verify | −148 [−183, −112] | better |
| C_costly_fail | +283 [+56, +510] | worse |
| D_weak_proposer | −157 [−207, −106] | better |
| E_gen_gap | −134 [−172, −96] | better |
| F_drift | −242 [−324, −160] | better |
| G_dear_judge | −118 [−160, −77] | better |
| H_noisy_judge | −132 [−185, −79] | better |
| J_no_judge | −19,401 [−19,476, −19,326] | better (the judge baseline collapses when a comparison costs 100) |

**Against the greedy judge, across 10 regimes:** better in 8, tied in 1, worse in 1.

**Against the best fixed rule in each regime (the fairer count):** better in 6 (base, B, D, E, F, G), tied in 3 (A, H, J), worse in 1 (C). In J the best fixed rule is the SRN-like note (ours −20 [−84, +45], tied), so J's −19,401 measures the judge baseline's failure, not an advantage of ours.

### Ablation (base regime; full controller = 623)
| Variant | Base Regret | Key Insight |
|---|---|---|
| Without propose gate | 694 | The gate is worth 71 units |
| Without verification channel | 671 | Paid probes add 48 units |
| Without probation channel | 700 | Live trials add 77 units |
| Without judge channel | 1,186 | Pairwise judgement is the most valuable channel (563 units) |
| Without any info channel | 1,177 | Buying info is worth 554 units; rule collapses to a fixed rule |
| 1-sigma acceptance margin | 1,515 | Risk aversion destroys the rule (rejects almost everything) |
| 2-sigma acceptance margin | 1,315 | As above |
| Flattened prior | 1,711 | Population prior carries most weight |

### Sweep (20 cells: verify_price × proposer_q, 20 seeds each)
- Controller wins **13 of 20 cells** (median margin −102 cost units vs the best fixed rule in the cell)
- Weak proposer (q = 0.25): controller loses to never updating (965 vs 574); its prior assumes proposer quality 1.0

## Real-LLM Replay (P3–P4): planned, not run
No language model has been called for this stage and no replay result exists. The design (`docs/06_replay_table_spec.md`):
- 5 task families × 40 tasks with hidden, learnable conventions and exact-match verifiers; designed empty-skill success ≈ 50%
- Two record types, counted separately: ~200 skills × 40 tasks = **8,000 agent-task cells**, plus ~**200 proposer calls**; with a ~150-call pilot, about **8,400 API calls** in total (~5 h at 30 requests/min)
- Pilot pass criteria: empty-skill success 25–60% on ≥4 of 5 families; valid proposer outputs with some candidates better and some worse than their parent; ≥95% determinism at temperature 0
- Every policy is then replayed from the table, with cost reported both in rollout-equivalents and in logged tokens

## Key Design Properties
- **Grid posterior** over D = v_cand − v_inc, 181 points, prior calibrated to EvoGuard (42.2% worse, 7.7% tie, 50% better)
- **Exact VoI**: enumerate all binomial outcomes, recompute the posterior, count gain only if the candidate would be accepted
- **Opportunity cost**: a live trial blocks the family's candidate slot; every consumed task is a lost update opportunity
- **Propose gate**: ask the proposer only when the best action on a fresh candidate is worth more than the proposal
- **Seven baselines**: never update, accept every, SRN-like note, greedy judge, threshold promote, probe gate, always verify

## Limitations
- Simulated stream: costs in rollout-equivalents, not real tokens or dollars
- No language model involved in any reported result
- Judge accuracy fixed at 0.7; real comparisons are noisier and biased
- Skill value is a scalar per family; context-dependent skills not represented
- Exponential tail shape is an assumption; lognormal sensitivity not yet run

## Files
- `paper.tex` at repository root → compiles with `pdflatex paper.tex` (run twice for references)
- Raw data is generated, not committed (`results/` is git-ignored). Regenerate and re-check with:
  ```sh
  uv run python experiments/run_grid.py --seeds 50   # results/grid.csv: 8,500 rows (~30 min)
  uv run python experiments/run_sweep.py --seeds 20  # results/sweep.csv
  uv run python experiments/verify_claims.py         # PASS/FAIL for every quoted number
  ```
- `docs/07_results_report.md`: results with figures and caveats
- `docs/04_simulator_spec.md`, `docs/06_replay_table_spec.md`: phase specifications
- Related work: EvoGuard 2611.06444, RSI Tax 2611.13524, FGSE 2611.14326, StepLearn 2609.35911, AtomSearch 2610.19951, SkillBoost 2607.26643, Skill∞ 2610.35434 (see `docs/01_literature_survey.md`)
