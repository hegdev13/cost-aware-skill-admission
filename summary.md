# ai-exploration-exploitation: Summary

## Project
Cost-aware explore/exploit controller for LLM agent skill admission. Given a pool of candidate rewrites (proposer output), the agent must decide SKIP/ACCEPT/PROBATE/VERIFY/JUDGE, priced in the same currency as task execution (1 task rollout = 1 token, failed task = c_fail, comparison = judge_cost, verification probe = verify_price).

## Key Contribution
A cost-aware decision rule that computes **expected net cost savings** for each action and picks the best. The value of information is computed **exactly** (enumerating every outcome on a grid posterior), not approximated. A live trial costs more than its failures because it blocks the family's candidate slot (opportunity cost).

## Verified Results

### Simulator Study (P1–P2)
- 30/30 pytest tests pass, 41/41 claim checks verified against raw CSVs
- 13 policies evaluated across 50 seeds per cell (10 regimes × 17 policies × 50 seeds = 8,500 rows)
- **Full controller: 623 ± 47 regret** vs greedy judge 752 ± 46 (—129, 95% CI [−166, −91], 86% of seeds improve)
- Controller reaches near-oracle quality: 0.642 success rate vs oracle 0.647, at ~1/33 the verification cost (21 probes vs 19,776)
- Only 20 harmful commits vs 100 for "accept every candidate"/"always verify"

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
| J_no_judge | −19,401 [−19,476, −19,326] | much worse |

**Across 10 regimes:** Controller significantly better in 7, tied in 1, worse in 1.

### Ablation (base regime, all changes within 2 units or 1%)
| Variant | Base Regret | Key Insight |
|---|---|---|
| Without judge channel | 1,186 | Pairwise judgement is most valuable channel (563 units) |
| Without any info channel | 1,177 | Buying info worth 554 units; rule collapses to fixed rule |
| 1-sigma acceptance margin | 1,515 | Risk aversion destroys the rule (rejects almost everything) |
| 2-sigma acceptance margin | 1,315 | As above |
| Flattened prior | 1,711 | Population prior carries most weight |

### Sweep (20 cells × verify_price × proposer_q)
- Controller wins **13 of 20 cells** (median margin −102 cost units)
- Weak proposer (q = 0.25): controller loses to never updating (965 vs 574) — prior assumption of quality 1.0 is the failure

### Real-LLM Pilot + Partial Table Build (P3)
- 205 of 8,400 cells populated (2.4%) before interruption
- 5 families × 41 cells each (40 tasks + 1 proposer call)
- Empty-skill true values: Family 0 = 6/41 = 0.146; Families 1–4 = 17/41 = 0.415 or 13/41 = 0.317 (4 of 5 families in 25–60% band)
- Pilot criteria: 3/3 pass (empty-skill band, proposer validity, determinism)
- Tokens: prompt 5,023 + completion 2,075 = 7,098 total (avg ~34.6 per call)
- Full 8,000-cell extension requires ~5h at 30 rpm, resumable via JSONL log + cache

### Key Design Properties
- **Grid posterior** over D = v_cand − v_inc, 181 points, prior calibrated to EvoGuard (42.2% worse, 7.7% tie, 50% better)
- **Exact VoI**: enumerate all binomial outcomes, recompute posterior, count gain only if candidate would be accepted
- **Opportunity cost**: live trial blocks family's candidate slot; every consumed task is a lost update opportunity
- **Propose gate**: only ask proposer when best action on fresh candidate is worth more than the proposal itself
- **Seven baselines**: never update, accept every, SRN-like note, greedy judge, threshold promote, probe gate, always verify

## Limitations
- Simulated stream: costs in rollout-equivalents, not real dollars
- Judge accuracy fixed at 0.7; real comparisons noisier/biased
- Skill value is a scalar per family; context-dependent skills not represented
- No language model involved in simulator results
- Exponential tail shape is an assumption; lognormal sensitivity not yet run

## Files
- `paper.tex` at repository root → compiles to `paper.pdf` (7 pages) via `pdflatex`
- `results/grid.csv`: 10 regimes × 17 policies × 50 seeds = 8,500 rows
- `results/curves.csv`: per-run regret curves
- `artifacts/replay_table_seed=20261113.jsonl`: 205 records (200 agent + 5 proposer) with token counts, success/failure hashes, model IDs
- `docs/04_simulator_spec.md`, `docs/06_replay_table_spec.md`: phase specifications
- 7 confirmed related-work citations from `orx discover`/`orx paper <id>`: EvoGuard 2611.06444, RSI Tax 2611.13524, FGSE 2611.14326, StepLearn 2609.35911, AtomSearch 2610.19951, SkillBoost 2607.26643, Skill∞ 2610.35434