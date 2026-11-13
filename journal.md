# Project journal

This is the working record of the project: why it exists, every design decision and the
reason for it, what is measured, what is logged, what has been run, what was found when the
project was checked, and what is meant to be shown. It is written so someone who was not in
the conversation can follow it. Facts here were checked against the code and the result files
on 2026-11-13; where something could not be checked it says so.

Related files: [README.md](README.md) (overview: what we are doing, what we found, what is
new), [setup.md](setup.md) (how to install, run and read the viewer),
[docs/01_literature_survey.md](docs/01_literature_survey.md),
[docs/02_research_plan.md](docs/02_research_plan.md),
[docs/03_outputs_and_presentation.md](docs/03_outputs_and_presentation.md),
[docs/04_simulator_spec.md](docs/04_simulator_spec.md),
[docs/05_decision_trace_and_ui_spec.md](docs/05_decision_trace_and_ui_spec.md),
[docs/06_replay_table_spec.md](docs/06_replay_table_spec.md),
[docs/07_results_report.md](docs/07_results_report.md).

---

## 1. The question, in plain words

An LLM agent improves between tasks by editing something external to the model: a skill
document, a memory, a note. After a task, a "proposer" (usually another LLM call) writes a
candidate edit. The agent has to decide what to do with it:

- **skip** it,
- **accept** it right away,
- put it on a **live trial** (use it on a share of real tasks and watch),
- pay for **verification** (extra held-out rollouts), or
- ask a cheap **pairwise judge** whether it beats the current skill.

Being wrong is not free. A bad skill makes real tasks fail, and every probe, proposal and
judgement costs tokens. This is the explore/exploit trade-off, but it sits **inside the
learning-update rule**: exploring means spending to learn whether a candidate is good,
exploiting means using what you currently believe.

**Research question.** Can a rule that keeps an explicit belief about each candidate and prices every
action (including buying information) spend less in total (failures plus tokens) than the
fixed rules used in the current literature, and in which cost regimes does it stop paying?

## 2. What is new (and what is not)

Checked against the literature in [docs/01](docs/01_literature_survey.md) (read in
November 2026; several neighbouring papers were only skimmed, marked `[abstract]` there):

- Fixed admission rules already exist: accept everything, greedy pairwise judge (EvoGuard),
  probe gate with confidence bounds (FGSE), promote after a fixed trial (StepLearn), one bounded
  note with no validation (SRN, from *The RSI Tax*).
- Bayesian value and bandits already exist for *retrieving* existing memories (BEAM, MEMOIR,
  OMCE) and for lifecycle management during RL training (TrackRL).
- **Not found in what I read:** admission treated as one sequential decision whose actions
  (skip / accept / trial / verify / judge / propose) are all priced in the same cost units, with the
  value of information computed exactly, and compared against oracle-referenced **cost-regret**
  across cost regimes.
- *The RSI Tax* shows the waste empirically (a simple bounded note matched costly methods at
  65.6 to 71.0% fewer tokens). It gives no decision rule. This project supplies one and tests it.

This is a claim about the literature I could read, not a proof that no one has done it.

## 3. Design decisions (chronological) and why

| # | Decision | Why | Evidence / where |
|---|---|---|---|
| D1 | Work on the *update rule* of an agent (cross-episode layer), not in-episode tool choice | The user asked for "the learning update rule with E/E in it"; the literature has hand-tuned gates there | docs/01 sections 2 and 5 |
| D2 | Frame it as cost-aware decision-making, not "a better bandit" | The strongest existing evidence (RSI Tax) is about cost, and fixed thresholds ignore it | docs/01 section 3 |
| D3 | Simulator first, real-LLM replay second | The simulator is cheap and gives the oracle (true skill values) that real systems never have; the replay then tests whether the logic survives real model behaviour | docs/02 section 5 |
| D4 | Common random numbers: all policies replay the same pre-drawn randomness per seed | Paired differences are much tighter than independent runs | `src/ee_update/sim.py` (`World`); test `test_crn_same_decisions_same_trajectory` |
| D5 | Report **cost-regret** against the cheaper of "never update" and a free-information oracle on the same seed | A policy that does nothing and a policy that knows everything both score 0, so the number cannot be gamed by either extreme | docs/04 section 6; `experiments/run_grid.py` |
| D6 | Calibrate candidate quality to EvoGuard's reported rewrite statistics (42.2% worse, 7.7% tie, 50% better; 7.7% lose and 11.3% gain at least 10 points) | It is the only measured distribution of single-trajectory skill rewrites that I found | docs/04 section 2.2; test `test_calibration_matches_table` |
| D7 | Derive exponential tail scales (0.0587 and 0.0672) from the two tail masses | The paper reports only the masses; the exponential shape is my assumption, so a heavier-tail check is still owed | `config.py`; arithmetic reproduced in the 2026-11-13 check |
| D8 | Improvements shrink near a ceiling (`v_max = 0.95`) | Without it every policy converges to the ceiling and a gate on proposing has nothing to decide late in the stream | amendments in docs/04 |
| D9 | Update opportunity every 5th task of a family | One proposal per task would make proposal cost dominate every comparison | `config.py` (`opp_every`) |
| D10 | One candidate in flight per family | Keeps the model small; makes a live trial block the proposer slot, which turned out to matter (D13) | `sim.py` |
| D11 | Baselines are fixed rules copied from papers, labelled "-style", never claimed as reproductions | The sources have different systems and benchmarks; only their decision rules and parameters are copied | `policies.py`; docs/03 table T2 |
| D12 | Controller belief is a **grid posterior** over D = v_cand - v_inc with a mixture prior, not a Gaussian | The prior has a point mass at "tie" and exponential tails; a Gaussian misprices information | `policies.py` (`D_GRID`, `_prior_log`); amendments in docs/04 |
| D13 | Value of information is **exact**: for a plan that buys n observations, every possible count is enumerated, the posterior recomputed, and the realised gain counted only if the candidate would then be accepted | No Gaussian shortcut in the one place it matters | `_trial_expected_gain`, `_judge_expected_gain` |
| D14 | A live trial is charged an **opportunity cost** | A trial holds the family's only candidate slot, so each task it consumes is an update opportunity in which no new candidate could be requested; without this the controller over-bought trials (ablation `c3_probe_only` isolates the effect) | `_scores`, `v_opp` |
| D15 | Add a **judge channel** to the controller | Removing it costs 537 regret units in the base regime (1,147 vs 610), the largest effect of any single channel | grid.csv; docs/07 section 4 |
| D16 | Gate on **proposing**: ask the proposer only when the best action on a fresh candidate is worth more than the proposal | Removing the gate costs 67 units in the base regime (677 vs 610) | `propose()`; policy `c3_voi_propose` vs `c1_voi` |
| D17 | The controller's prior comes from the default `Config()`, not the run's true config | So regimes that change the proposer (D) or the verification gap (E) test **misspecification** instead of handing the controller the answer | `BeliefController.__init__` (`self.A = Config()`) |
| D18 | Controller is **risk-neutral** (acceptance margin z = 0); risk-averse variants exist only as ablations | A 1-sigma or 2-sigma margin makes it accept nothing (docs/07 section 4; verification log below) | `c1_conservative`, `c1_risk_averse` |
| D19 | One append-only log per run; HTML and terminal viewers render only that log | So what is shown is exactly what the system did | `trace.py`, docs/05 |
| D20 | Truth (hidden values) lives in a separate layer, written to `<policy>.truth.jsonl` | The policy never sees it; the viewer can hide or reveal it | `trace.py` (`log_truth`) |
| D21 | Two viewers: offline single-file HTML and a Textual terminal app | The user asked for both; the HTML opens by double-click for an advisor meeting | `experiments/build_viewer.py`, `src/ee_update/tui.py` |
| D22 | Real-LLM stage uses a replay table built once, then replayed by every policy | Real outcomes, identical randomness, no repeat API cost per policy, ground-truth values for regret | docs/06 (not started) |
| D23 | Open models on the user's NVIDIA key; self-imposed rate cap (30 requests/min, concurrency 4, 12,000 calls, 20M tokens) | The user said the models are free; the cap is a conservative guess, not a documented limit | docs/06 section 6 |
| D24 | The value-of-information enumeration uses the binomial pmf (2026-11-14) | The earlier code dropped the binomial coefficient and normalised each grid point by its own maximum, so D13's "exact" was false; fixing it changed the base regret 623 -> 610 and turned regime C from a loss into a tie | `_trial_expected_gain`, `_binom_pmf`; `test_trial_value_of_information_is_exact` |
| D25 | Real-LLM table on Google Gemini (free tier) instead of NVIDIA-hosted open models | The NVIDIA endpoint timed out; the free tier allows 15 requests/minute and 500/day per model, which forced 10 tasks per agent call and 9 candidates per family | `experiments/build_replay_table.py`; docs/06 section 11 |
| D26 | The proposer is the agent's own model; the judge is a different model | The agent rewriting its own notes is the self-improvement setting under study; a separate judge avoids grading its own work | `AGENT_MODEL`, `PROPOSER_MODEL`, `JUDGE_MODEL` |
| D27 | Replay at two price settings: the simulator's assumed prices and prices measured from the recorded tokens | The assumed judge price (0.1 rollout) turned out to be 47x too low; the controller reads prices, fixed rules do not, so measured prices are the fair test | `ReplayTable.measured_prices()`; `run_replay.py` |

## 4. How the world works (as built)

All of this is in `src/ee_update/config.py` and `sim.py`. Units: **one task rollout = 1**.

- **Stream.** 6 task families, 1,000 tasks, family drawn uniformly. Each family has one incumbent
  skill with a hidden success probability `v` (initial `v` ~ Normal(0.5, 0.15) clipped to [0.05, 0.9]).
  A task succeeds with probability `v` of the skill used.
- **Candidates.** A proposal creates a candidate whose value is the parent's plus `delta`. `delta` is
  worse (probability 0.422, exponential scale 0.0587), a tie (0.077) or better (0.500, exponential scale
  0.0672 scaled by remaining headroom). The proposer-quality knob `q` multiplies the "better" share.
- **Costs.** A failed task costs `c_fail = 5` extra. A proposal costs 1. A held-out verification probe
  costs `verify_price = 1` each. A pairwise judge costs `judge_cost = 0.1` and is right with
  probability `judge_acc = 0.7`.
- **Observation channels the policy has.**
  1. Live outcomes of whatever skill served each task.
  2. **Probation**: a share `q_p = 0.2` of the family's tasks goes to the candidate.
  3. **Verification**: k held-out probes of the candidate, optionally offset by a per-candidate
     generalisation gap `shift_sd`.
  4. **Judge**: a binary "better / not better" verdict.
- **What a policy never sees:** the true values, the candidate's `delta`, drift, the shift. A policy gets a
  frozen `View` of observable data only (test `test_views_hide_truth`).
- **Regimes** (overrides on the defaults): base; A cheap verification (0.2 per probe); B costly
  verification (2.0); C costly failures (`c_fail = 20`); D weak proposer (`q = 0.5`); E generalisation
  gap (`shift_sd = 0.08`); F drift (`drift_sd = 0.01`); G dear judge (1.0); H noisy judge (accuracy
  0.55); J no judge (cost 100).

## 5. The policies

| Policy | Rule | Source of the rule |
|---|---|---|
| `static` | never propose | control |
| `accept_all` | propose every opportunity, accept unchecked | common baseline |
| `srn_like` | propose only after a failed task, overwrite, no validation | *The RSI Tax* (SRN) |
| `greedy_judge` | one pairwise judgement, accept the winner, never revisit | EvoGuard |
| `threshold_promote` | live trial for 10 tasks, accept if its win rate beats the incumbent's | StepLearn |
| `probe_gate` | 20 held-out probes, accept if the gain's lower confidence bound (z = 1.645) is at least 0.03 | FGSE |
| `always_verify` | 100 held-out probes, accept if the point estimate beats the incumbent | AtomSearch-style cost |
| `c1_voi` | belief controller, proposes every opportunity | this work |
| `c3_voi_propose` | `c1_voi` plus the gate on proposing (**"ours"**) | this work |
| ablations (7) | no information bought; no verify channel; no probation channel; no judge channel; 1-sigma and 2-sigma margins; flattened prior | this work |
| `oracle` | sees true values, tests for free, proposes only while expected gain exceeds the proposal cost | reference only |

**How the controller decides** (in cost units, relative to "keep the incumbent"; `H` = remaining
tasks of the family, `c = c_fail`):

- skip = 0
- accept = `c * H * mu`, where `mu` is the posterior mean advantage of the candidate
- trial of n observations = `c * (H - n/q_p) * E[D * 1{accept after the data}] + n * c * mu - (lost
  proposal opportunities) * (value of an opportunity)`
- verify k probes = `c * H * E[D * 1{accept after the data}] - k * verify_price`
- judge = `c * H * E[D * 1{accept after the verdict}] - judge_cost`

It takes the maximum. Each decision record stores all five scores, the greedy action
(what it would have done without buying information) and a one-sentence reason.

## 6. Metrics (precise definitions)

All come from `Simulator.run` and `experiments/run_grid.py`.

- `total_cost` = `c_fail * failures + task_rollouts + proposal + verification + judge`.
- `regret` = `total_cost` minus the cheaper of `static` and `oracle` on the same seed. Zero means "as cheap
  as doing nothing or knowing everything".
- `success_rate` = 1 minus failures divided by tasks. `final_value` = mean true value of the final
  incumbents.
- `proposals`, `accepts`, `rejects`, `explore_actions` (probation, verification, judge and proposals count
  as explore).
- `harmful_accepts` = accepted candidates whose true `delta` was not positive (ties count as harmful).
  `rejected_good` = rejected candidates whose true `delta` was positive.
- `rejected_tokens` = proposal, verification and judge spend on candidates that were never accepted.
- **Paired comparison**: per-seed differences between two policies (same seeds, same randomness), mean with a
  95% t interval (`experiments/summarize.py`, function `paired`), plus the share of seeds where the first policy
  is cheaper.
- **Explore / exploit label** (docs/05): a decision is EXPLORE if it buys information (trial, verify, judge, or a
  proposal) and EXPLOIT otherwise. The record also stores the greedy action.

## 7. What the logs contain

Each run can write `<policy>.trace.jsonl` (the log the policy produced) and `<policy>.truth.jsonl` (hidden
values, oracle-only). One JSON object per line, ordered by `seq`.

| `type` | Fields beyond `seq`, `t`, `family` | Meaning |
|---|---|---|
| `META` (first line) | `policy`, `seed`, `config` | run identity and full configuration |
| `TASK_SERVED` | `skill_id`, `routing_rule`, `arm`, `outcome`, `tokens`, `cost` | which skill served a task and why (the "skill invoked" record) |
| `PROPOSE_DECISION` | `chosen`, `mode`, `rule_id`, `rule_inputs`, `scores`, `reason` | whether to ask the proposer |
| `SKILL_CREATED` | `skill_id`, `parent_id`, `proposer_call_id`, `source`, `tokens` | a candidate exists (the simulator has no skill text; text diffs appear in the replay stage) |
| `ACTION_CHOSEN` | `candidate_id`, `action`, `k`, `mode`, `greedy_action`, `rule_id`, `rule_inputs`, `scores`, `reason` | the core decision record |
| `PROBATION_UPDATE` | `candidate_id`, `n_cand`, `wins_cand`, `n_inc`, `wins_inc`, `target_obs`, `posterior` | live-trial progress |
| `VERIFY_RESULT`, `JUDGE_RESULT` | probes or verdict, `tokens` | information that was bought |
| `SKILL_ACCEPTED`, `SKILL_REJECTED` | `skill_id`, `rule_id`, `rule_inputs`, counts | the decision was applied |
| `COST_CHARGED` | `component`, `amount`, `caused_by`, `skill_id` | ledger entry |

Contract tests (`tests/`): ledger sums equal the run total; every accept or reject follows an
action on the same skill; every served skill was created; no `TRUTH` inside the policy-visible log;
the HTML and terminal viewers agree on counts; and a real-browser smoke test (see section 9).

**Reading one decision record.** Take candidate `f5-c9` at task 304 in the base-regime trace of seed 3
(file `results/viewer/c3_voi_propose.trace.jsonl`). Its belief says the candidate looks slightly worse
(`mu_D = -0.016`, `P(better) = 0.31`), the scores are skip 0.0, accept -9.2, **judge +2.3**, probation -24.7,
verification -5.0, so the controller pays 0.1 for a judgement. The greedy action would have been
SKIP, so the card shows "this decision buys information". (One viewer quirk is described in section 9.)

## 8. Experiments that were run

| Experiment | What | Size | Output | Cost |
|---|---|---|---|---|
| Unit and contract tests | simulator invariants, calibration, accounting, trace contract, viewers | 54 tests | pytest | about 10 s |
| **Grid** (`experiments/run_grid.py`) | 10 regimes x 17 policies x 50 seeds, T = 1,000, seeds 0 to 49 | 8,500 rows | `results/grid.csv`, `results/curves.csv` | 1,800 s recorded in `grid_meta.json` |
| **Sweep** (`experiments/run_sweep.py`) | verification price (5 values) x proposer quality (4 values) x 8 policies x 20 seeds | 160 cells | `results/sweep.csv` | 595 s recorded in `sweep_meta.json` |
| Summaries and figures | tables T1 to T3, six figures | - | `results/tables.md`, `figs/` | seconds |
| Viewer build | seed 3, base regime, six policies with full traces | 6 traces | `results/viewer/` | about 1.5 s |
| **Claim check** (`experiments/verify_claims.py`) | re-derives every number quoted in docs/07 and this journal from the raw data | 52 checks (41 on 2026-11-13) | PASS/FAIL lines, exit code | about 2 s |
| **Margin check** (`experiments/tune_margin.py`) | acceptance margin z x gate on proposing, over the four development regimes only | 8 settings x 4 regimes x 25 seeds | `results/tuning.csv` | 634 s |

| **Replay table** (`experiments/build_replay_table.py`, 2026-11-14) | 5 families x (empty skill + 9 candidates) x 40 tasks; 45 proposer and 450 judge calls on Gemini | 2,000 cells | `data/replay/` (committed) | about 700 API calls on the free tier |
| **Replay** (`experiments/run_replay.py`) | 17 policies x 50 seeds x 600 tasks at assumed and at measured prices | 1,700 runs | `results/replay_*.csv`, `replay_tables.md`, `replay_summary.json` | about 2 min |

Until 2026-11-13 no language model was used anywhere; the replay table was built on 2026-11-14 (section 9c).

## 9. Verification log (2026-11-13)

What I did to check the project, and what I found. Nothing below was assumed from earlier summaries.

**Reproduced exactly.**
- Test suite: 30 pass on the working tree and on a fresh clone (`git clone`, `uv sync`, `uv run pytest -q`).
- Second pass, same day, after the documentation was written: the whole suite still passes (30), and
  `experiments/verify_claims.py` now re-derives 41 claims from the raw CSVs (headline table, paired regime
  comparison, ablation numbers, sweep claims, the quoted decision record, the recorded run metadata) and all
  41 pass. That script is the reproducible half of this section; re-run it after any change to the data.
- The trace files under `results/viewer/` were rebuilt after the controller changed (the live-trial
  opportunity cost), so the decision record quoted below is the one in the file on disk today
  (candidate `f5-c9`, task 304, `seq` 847). A candidate can be decided several times within one update
  opportunity, so the same candidate and task appear more than once in the log; the quoted record is
  the one whose posterior mean is -0.016.
- 12 of 12 re-run (regime, policy, seed) cells match `results/grid.csv` to 1e-6 (base, D, F, plus the oracle).
- The base-regime table in docs/07 (regret, cost, failures, proposals, verification, judge, success, harmful
  commits for every policy) matches the CSV.
- Sweep claims: controller wins 13 of 20 cells, median margin -102 across all cells; weak proposer
  (q = 0.25) loses to never updating (965 vs 574 at verification price 0.5).
- Ablation numbers (694, 671, 700, 1,186, 1,177, 1,515, 1,315, 1,711) match the CSV; conservative variants
  accept 0.0 candidates on average in the base regime.
- Headline base-regime paired difference -128.6 vs greedy judge (the report says -129); controller better on
  86% of seeds.

**Defects found and fixed.**
1. **The HTML viewer rendered an empty page** (policy menu blank, no data). A script error
   (`NAMES.forEach is not a function`: the payload's `policies` is an object, the code treated it as an
   array). The Python tests could not see it. Fixed in `experiments/build_viewer.py`; a headless-Chrome test
   now fails if the page throws or does not draw decision cards and the ledger. I also drove the page
   (scrubbing, truth layer, compare mode, tabs, policy switching, play) with no errors.
2. The ledger line "reference (free-information oracle)" never showed because the oracle run was not in the payload;
   it now shows the cheaper of never-updating and the oracle. The scrubber thumb also started at 10%; fixed.
3. **Regime counts in the report were wrong.** The report said "seven better, one tied, one worse" (sums to 9) and
   the commit title says 7 of 10. Recomputed from the CSV: against the greedy judge, **8 better, 1 tied (A), 1 worse
   (C)**; against the best fixed rule in each regime, **6 better, 3 tied (A, H, J), 1 worse (C)**. The report is corrected.
4. The J regime's headline gap (-19,401) is not an advantage of ours: the greedy judge is the wrong comparator when
   a judgement costs 100, and a simple fixed rule is as good as the controller there. The report now says so.
5. The report explained the failure of risk-averse variants with "the prior's uncertainty is about 0.09". The margin
   is applied to a wider spread that also includes uncertainty about the incumbent's own value (median 0.16 on the
   seed-3 trace, versus 0.08 from the candidate posterior alone and a posterior mean that reaches at most 0.034).
   The conclusion holds a fortiori; the text is corrected.
6. `uv run python -m ee_update.tui` failed on one machine state with `No module named 'ee_update'` (the editable
   install was missing its path file). `uv sync --reinstall-package ee-update` fixes it. A fresh clone works.

**Known quirks, not changed.**
- After a candidate is accepted with no live data (for example after a judge acceptance), the controller's belief
  about the new incumbent falls back to its broad prior (mean 0.5, sd 0.25) until live outcomes accrue. This does
  not change the grid posterior used to decide (it enters only through the displayed spread and the ablation
  margins), but the "belief sd" shown on cards (`s_D`) is larger than the candidate's own posterior sd; 9% of
  decisions on the seed-3 trace had incumbent sd of at least 0.2.
- The HTML file embeds the truth layer; the toggle only hides it. It is a presentation control, not a security boundary.

**The one hyperparameter choice, now reproducible.** The controller has two hand-set
parameters: the acceptance margin `z` and whether to gate proposing. `experiments/tune_margin.py`
sweeps both over the four **development** regimes (base, D_weak_proposer, F_drift, G_dear_judge)
and leaves the other six regimes untouched. Result (mean regret over the four regimes, 25 seeds):
z = 0 with the gate 709, z = 0 without the gate 773, z = 0.15 with the gate 1,383, z = 0.3 1,370,
z = 0.5 1,373 (re-run 2026-11-14 with the exact enumeration; on 2026-11-13 it was 716 / 782 / 1,403 / 1,392 /
1,459, the same choice). So the shipped controller uses **z = 0 and the gate**, and the choice was made
without looking at A, B, C, E, H or J. Any future change to `z` or to the gate must be re-checked
this way.

**Not verified.**
- The literature numbers copied from other papers were read from their text during the survey but not re-read
  today.
- The EvoGuard-calibrated tail shape (exponential) has not been tested against a heavier tail.
- The claim in docs/07 section 7 that the controller's hyperparameters were chosen on the development
  regimes was not confirmable from history at the time of writing; it is now settled by
  `experiments/tune_margin.py` above (see the verification log).

## 9b. The documentation set

| File | Purpose | Audience |
|---|---|---|
| `README.md` | the whole project in one page: the question, what was built, what was found, what is new, how to run it, the file map, next steps | anyone landing on the repository |
| `journal.md` (this file) | the working record: every design decision and why, the world model, the policy rules, metric definitions, the log schema, the experiments, the verification log, what to present, what is left | the next person to work on it |
| `setup.md` | install, run, read the viewer, reproduce the experiments, extend the project, troubleshoot | whoever has to run it |
| `docs/07_results_report.md` | the results with figures and the limitations | the paper's results section |

The three top-level files deliberately overlap: `README.md` is the pitch, this journal is the
evidence, `setup.md` is the manual. Where they disagree, the code and the CSVs win.

## 9c. Verification log (2026-11-14): exact VoI, and the first real-LLM replay

- **VoI bug.** A review found that `_trial_expected_gain` was not the exact enumeration D13 claims (for two coin
  flips at p = 0.5 it predicted counts 1/3, 1/3, 1/3 instead of 1/4, 1/2, 1/4). Fixed (D24) and tested against a
  brute-force loop. Before re-running, a paired 50-seed check per regime showed the fix never hurts: within noise
  in nine regimes, -168 +/- 158 in C.
- **Refactor check.** The simulator's outcome draws moved into hooks (`_task_success`, `_make_candidate`,
  `_verify`, `_judge`, `_make_root`); 64 baseline runs reproduce the previous grid's costs exactly.
- **Full re-run** of the grid and the sweep with the fixed controller; every quoted number updated (README,
  docs/07, paper, summary). Changes: base 610 (-142 [-179, -105] vs greedy judge); vs the judge 8 better / 2
  tied / 0 worse; vs the best fixed rule per regime 6 / 4 / 0 (now checked by `verify_claims.py`, previously
  only quoted); sweep 14 of 20 cells, median -112; ablations 677 / 671 / 683 / 1,147 / 1,177 / 1,286 / 1,285 /
  1,733. The quoted decision record (f5-c9, t = 304) still exists, with PROBATE -24.7 and VERIFY -5.0.
- **Earlier replay attempt rejected.** An uncommitted attempt (205 rows) in another worktree could not be used:
  its agent never received a skill, its proposer read the string "WRONG_ANSWER" instead of real answers, a
  proposer row was counted as a task success in every family's rate, the "3/3 pilot criteria" came from a mode
  that makes no API call, token counts were estimated from characters, and rows said `google/gemma-3-4b-it`
  while the calls went to Gemini. The task suite was kept, after fixing reversed unit conversions.
- **Replay table built** (D25-D27): 2,000 agent-task cells, 45 proposer calls, 450 judge calls, all with the
  service's own token counts; committed in `data/replay/`. Empty skills 0.40-0.425 (pilot criterion A passes);
  every candidate at least ties its parent (43 better, 2 tied).
- **Replay results** (50 seeds): at assumed prices ours 54, best fixed rule (SRN-like) 91, -37 [-62, -12]; at
  measured prices ours 281 vs SRN-like 85, +196 [+150, +241]: the propose gate stops after 4.5 proposals because
  the EvoGuard prior undervalues rewrites that are really worth about 45 points.
- `verify_claims.py`: 52/52 (10 new replay claims). Test suite: 54 pass.

## 10. What is to be presented

**One-sentence message.** A rule that prices every way of finding out whether a candidate skill is good
spends less than the fixed rules in most cost regimes, never loses to the best fixed rule of a regime (6 better,
4 tied), fails in one identifiable corner (a weak proposer), and every decision it makes can be read from a log.

**Five-minute advisor walk-through.**
1. Open `results/viewer/viewer.html` (build it first, see [setup.md](setup.md)). Pick `c3_voi_propose`.
2. Read the task strip: dots are tasks, triangles are created candidates, circles accepted, squares rejected.
3. In the decision cards find an EXPLORE card (a judge or verification). Show the five scores and the greedy
   action. Turn on **truth layer** to show whether buying the information paid off.
4. Switch **compare** to `accept_all` or `greedy_judge`: same stream, same candidates, different decisions and
   a different cost ledger.
5. Show `figs/fig1_regret_curves.svg`, then `figs/fig4_regime_map.svg` (where it wins and where it does not),
   then `figs/fig6_ablations.svg` (which channel matters).

**Figures and what each answers** (all rebuilt from the CSVs by `experiments/make_figures.py`):
fig1 regret over the stream; fig2 where the cost goes; fig3 accuracy against tokens; fig4 the regime map;
fig5 one candidate's decision path; fig6 ablations.

**Say these limitations out loud.** The simulator involves no language model and the real-LLM replay is small
(five synthetic families, 45 candidates, one model family, free-tier sized); at measured prices the controller
loses to a bounded note on that replay; the candidate-quality model is calibrated to one paper's single benchmark; one scalar value per skill and one candidate at a time; the judge is a
fixed-accuracy sign; the controller's prior is fixed and it loses to "never update" with a weak proposer;
it is risk-neutral and only ties when failures are very costly (regime C).

## 11. What is left

1. **Learn the proposer online** so the weak-proposer corner of the regime map stops losing to "never update".
2. **Scale risk with stakes** (regime C is now a tie with the greedy judge, +115 [-121, +351]; a stake-scaled margin may turn it into a win).
3. **A bigger real-LLM replay** (docs/06 section 11 lists what the first one cut): harder families where
   rewrites can hurt, a skill tree, one call per task, more models. Needs `GEMINI_API_KEY` (or a paid tier).
4. **Heavier-tailed candidate quality** and the held-out-regime check.
5. Re-read the abstract-only neighbours in the survey before writing related work.
