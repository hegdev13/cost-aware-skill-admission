# ai-exploration-exploitation

**Explore/exploit inside the learning-update rule of LLM agents.** When a proposer offers an
agent a candidate rewrite of its skill or memory, the agent has to decide what to do with it:
skip it, accept it, put it on a live trial, pay for verification, or ask a cheap pairwise
judge. Getting it wrong is not free - a bad skill makes real tasks fail, and every probe,
proposal and judgement costs tokens. This project builds a rule that keeps an explicit
belief about each candidate and prices every action, including buying information, and then
measures where that rule wins and where it loses.

Status: the simulator stage is **complete and verified** (30 tests, 41 claims checked against
the raw data). The real-LLM replay stage is specified but not built. Everything runs on a
laptop CPU; no GPU, no API key, no network needed.

## The question

An agent that learns between tasks edits something outside the model: a skill document, a
memory, a note. Fixed rules do this today - accept everything, ask one pairwise judge and
take the winner, run a fixed number of probes and commit if the lower confidence bound
clears a bar, overwrite a bounded note after a failure. None of them ask what the test is
worth. This project asks: **can a rule that prices every way of finding out whether a
candidate is good spend less in total (failures plus tokens) than those fixed rules, and in
which cost regimes does it stop paying?**

## What we built

**A controller** ("ours") that, for each candidate skill:

1. keeps a **belief** about the candidate's advantage over the incumbent - a grid posterior
   whose prior is the population of rewrite deltas calibrated to published rewrite statistics
   (42.2% worse, 7.7% tie, 50% better), updated by the candidate's own live outcomes, its
   paid verification rollouts and pairwise judge verdicts;
2. prices five actions in the same cost units - skip (0), accept, live trial of n
   observations, verification of k probes, one judge call - and takes the best;
3. computes the **value of information exactly**: for a plan that buys n observations, every
   possible outcome is enumerated, the posterior is recomputed, and the realised gain is
   counted only if the candidate would then be accepted. No Gaussian shortcut;
4. charges a live trial for the **update opportunities it blocks** (a trial holds the
   family's only candidate slot), and only asks the proposer when the best action on a fresh
   candidate is worth more than the proposal.

**Seven fixed baselines** with their rules and parameters copied from the literature
(EvoGuard-style greedy judge, StepLearn-style threshold promotion, FGSE-style probe gate,
SRN-style bounded note from *The RSI Tax*, always-verify, accept-all, never-update), a
**free-information oracle**, and seven **ablations** of our own rule.

**A cost-regret metric** against the cheaper of never-updating and the oracle on the same
seed, so "do nothing" and "know everything" both score 0, and every policy sees the same
pre-drawn randomness per seed, so comparisons are paired.

**Two viewers over one log.** Every run appends an event log; the offline HTML page and the
terminal UI only render that log, so what you show an advisor is exactly what the system did.
Each decision card carries the belief, the net value of every alternative including the one
not taken, the greedy action (what it would have done without buying information), and a
one-sentence reason - with the hidden truth in a separate layer you can reveal afterwards.

## What we found

Base regime (6 task families x 1,000 tasks, 50 seeds, cost in rollout-equivalents):

| policy | cost-regret | total cost | failures | proposals | verification | judge | final success | harmful commits |
|---|---|---|---|---|---|---|---|---|
| never update | 1,248 | 3,571 | 514 | 0 | 0 | 0 | 0.486 | 0 |
| accept every candidate | 1,291 | 3,614 | 483 | 198 | 0 | 0 | 0.517 | 100 |
| SRN-like bounded note | 1,127 | 3,450 | 472 | 92 | 0 | 0 | 0.528 | 47 |
| threshold promote (StepLearn-style) | 1,151 | 3,474 | 490 | 22 | 0 | 0 | 0.510 | 3 |
| greedy pairwise judge (EvoGuard-style) | 752 | 3,075 | 371 | 198 | 0 | 20 | 0.629 | 34 |
| probe gate with LCB (FGSE-style) | 5,070 | 7,392 | 448 | 198 | 3,955 | 0 | 0.552 | 2 |
| always verify (AtomSearch-style cost) | 20,417 | 22,740 | 353 | 198 | 19,776 | 0 | 0.647 | 20 |
| **ours** | **623** | **2,946** | 358 | 115 | 21 | 21 | 0.642 | 20 |
| free-information oracle | 0 | 2,323 | 236 | 141 | 0 | 0 | 0.764 | 0 |

- Against the strongest fixed rule (the greedy judge): **-129 [-166, -91]** regret units,
  better on **86% of seeds**. Same skill quality, 14 fewer harmful commits, 83 fewer
  proposals, 13 fewer failures.
- Across ten named cost regimes: **8 better, 1 tied, 1 worse** than the greedy judge. The
  one loss is the regime where a failed task costs 20 rollout-equivalents.
- In a price x proposer-quality sweep (20 cells): the controller wins **13 of 20**, median
  margin -102 units.
- The two fixed-probe rules reach the *best* skill quality of all and pay 20,000
  rollout-equivalents for it - the RSI Tax result reproduced inside a controlled stream.
- The controller lands 27% above the free-information oracle's cost, which is the price of
  not knowing the true values.
- What each ingredient is worth (base regime): the judge channel 563 units, buying any
  information 554, the trial 77, the propose gate 71, verification 48. A 1-sigma
  acceptance margin *costs* 892 units: the rewrite effect sizes are smaller than the
  uncertainty they are applied to, so a conservative gate accepts nothing.

Two things we did not expect, both recorded in the journal: a live trial is dearer than it
looks (it blocks future proposals, which the model found and the ablation confirmed), and a
risk-averse acceptance rule is worse than a risk-neutral one in this population.

Where it fails, and why it matters: with a **weak proposer** the controller loses to "never
update" (965 vs 574) because its prior about the population of rewrites is fixed at the
published calibration. That is a *learning* failure, not a decision failure, and the fix -
estimate the proposer's quality online from candidates already tried - is the next step.

## What is new

Checked against the literature in `docs/01_literature_survey.md` (read November 2026; some
neighbours were only skimmed and are marked `[abstract]` there):

- Fixed admission rules already exist; Bayesian value and bandits already exist for
  *retrieving* memories and for RL lifecycle management. What I did not find in what I read
  is admission treated as one sequential decision whose actions - including buying
  information - are priced in the same cost units, with the value of information computed
  exactly and compared against an oracle-referenced cost-regret across cost regimes.
- *The RSI Tax* shows the waste empirically (a simple bounded note matched costly methods at
  65.6-71.0% fewer tokens) but gives no decision rule. This project supplies one and tests
  it.
- This is a claim about the literature I could read, not a proof that no one has done it.

## Run it

```sh
git clone --branch ee-update-rule <repo-url> ee-update && cd ee-update
uv sync
uv run pytest -q                                   # 30 tests, about 10 s
uv run python experiments/build_viewer.py          # build the offline HTML viewer
open results/viewer/viewer.html                    # the interactive trace
```

Full instructions, including the terminal viewer, the experiments and troubleshooting, are in
**[setup.md](setup.md)**. The working record - every design decision and why, the metrics,
the log schema, what was run, what was found, and what is left - is in
**[journal.md](journal.md)**. The results with figures and caveats are in
**[docs/07_results_report.md](docs/07_results_report.md)**.

## The files

| Path | What it is |
|---|---|
| **[journal.md](journal.md)** | the working record: design decisions D1-D23 with reasons, world model, policy rules, metric definitions, log schema, experiments run, verification log, what to present, what is left |
| **[setup.md](setup.md)** | how to install, run, read the viewer, reproduce the experiments, extend the project |
| `docs/01_literature_survey.md` | what exists in the literature, what is covered, where the gap is |
| `docs/02_research_plan.md` | now vs target, research questions, phases and their exit criteria, the kill criterion |
| `docs/03_outputs_and_presentation.md` | the figure and table plan, possible outcomes |
| `docs/04_simulator_spec.md` | the simulator design and its calibration, plus the amendments that record where the build differs from the spec |
| `docs/05_decision_trace_and_ui_spec.md` | the event log, how explore/exploit is labelled, the two viewers |
| `docs/06_replay_table_spec.md` | the real-LLM replay table, task suite, models, rate cap (not built) |
| `docs/07_results_report.md` | the results, the regime map, the ablations, the limitations |
| `src/ee_update/config.py` | every parameter and the ten named regimes |
| `src/ee_update/sim.py` | the simulator and the decision loop |
| `src/ee_update/policies.py` | baselines, oracle, belief controller, ablations |
| `src/ee_update/trace.py` | the append-only decision log |
| `src/ee_update/tui.py` | the terminal viewer |
| `experiments/` | grid, sweep, summaries, figures, viewer builder, claim verifier, artifact publisher |
| `tests/` | 30 tests: simulator invariants, calibration, accounting, trace contract, viewers |
| `figs/` | the six publication figures (PDF and SVG) |
| `results/` | generated CSVs, traces and the HTML viewer (git-ignored; rebuild with the scripts above) |

## Reproducing the numbers

```sh
uv run python experiments/run_grid.py --seeds 50     # 10 regimes x 17 policies x 50 seeds, about 30 min
uv run python experiments/run_sweep.py --seeds 20    # price x proposer-quality sweep, about 10 min
uv run python experiments/summarize.py               # writes results/tables.md
uv run python experiments/make_figures.py            # writes figs/fig1..fig6
uv run python experiments/verify_claims.py           # re-checks every number quoted in the docs
uv run python experiments/publish_artifacts.py       # copies report, figures, data, viewer to the artifacts folder
```

`verify_claims.py` is the reproducible half of the verification log in `journal.md`: it
re-derives the headline table, the paired regime comparison, the ablation numbers, the sweep
claims and the quoted decision record from the raw CSVs and prints PASS/FAIL for each.

## Next steps

1. **Learn the proposer online** so the weak-proposer corner of the regime map stops losing
   to "never update" - the update rule learning its own environment.
2. **Scale risk with the stakes** to recover the costly-failure regime without breaking the
   base regime.
3. **Real-LLM replay** (`docs/06_replay_table_spec.md`): a pilot, then a replay table built
   with open models on an NVIDIA key, then every policy replayed on real outcomes and shown
   in the same viewers. The simulator says which decisions to look at; the replay says
   whether the logic survives real model behaviour.
4. Heavier-tailed candidate quality, and the held-out-regime check of the controller's
   hyperparameters.
