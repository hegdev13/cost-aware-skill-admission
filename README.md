# ai-exploration-exploitation

**Explore/exploit inside the learning-update rule of LLM agents.** When a proposer offers an
agent a candidate rewrite of its skill or memory, the agent has to decide what to do with it:
skip it, accept it, put it on a live trial, pay for verification, or ask a cheap pairwise
judge. Getting it wrong is not free - a bad skill makes real tasks fail, and every probe,
proposal and judgement costs tokens. This project builds a rule that keeps an explicit
belief about each candidate and prices every action, including buying information, and then
measures where that rule wins and where it loses.

Status: the simulator stage is **complete and verified**, and a first **real-LLM replay** has
been built and run (54 tests; 52 quoted numbers re-checked against the raw data by
`experiments/verify_claims.py`). Everything runs on a laptop CPU; the recorded real-LLM table
is committed, so no API key is needed unless you rebuild it.

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
| **ours** | **610** | **2,932** | 358 | 116 | 7 | 21 | 0.642 | 20 |
| free-information oracle | 0 | 2,323 | 236 | 141 | 0 | 0 | 0.764 | 0 |

- Against the strongest fixed rule (the greedy judge): **-142 [-179, -105]** regret units,
  better on **86% of seeds**. Same skill quality, 14 fewer harmful commits, 82 fewer
  proposals, 13 fewer failures.
- Across ten named cost regimes: **8 better, 2 tied, 0 worse** than the greedy judge; against
  the best fixed rule *of each regime*, 6 better, 4 tied, 0 worse. The closest call is the
  regime where a failed task costs 20 rollout-equivalents (+115 [-121, +351], a tie).
- In a price x proposer-quality sweep (20 cells): the controller wins **14 of 20**, median
  margin -112 units.
- The two fixed-probe rules reach the *best* skill quality of all and pay 20,000
  rollout-equivalents for it - the RSI Tax result reproduced inside a controlled stream.
- The controller lands 26% above the free-information oracle's cost, which is the price of
  not knowing the true values.
- What each ingredient is worth (base regime): buying any information 567 units, the judge
  channel 537, the trial 73, the propose gate 67, verification 61. A 1-sigma
  acceptance margin *costs* 676 units: the rewrite effect sizes are smaller than the
  uncertainty they are applied to, so a conservative gate accepts nothing.

Two things we did not expect, both recorded in the journal: a live trial is dearer than it
looks (it blocks future proposals, which the model found and the ablation confirmed), and a
risk-averse acceptance rule is worse than a risk-neutral one in this population.

Where it fails, and why it matters: with a **weak proposer** the controller loses to "never
update" (962 vs 574) because its prior about the population of rewrites is fixed at the
published calibration. That is a *learning* failure, not a decision failure, and the fix -
estimate the proposer's quality online from candidates already tried - is the next step.

### On real model outputs

A replay table of real Gemini outcomes (`data/replay/`: 5 task families x 40 tasks, 45
candidate skills written by the model from its own failures, 2,000 recorded agent-task
outcomes, 450 judge verdicts), with every policy replayed on it over 50 seeds:

- **At the simulator's prices the controller wins on real outcomes**: lowest regret of every
  non-oracle policy, -37 [-62, -12] against the best fixed rule (an SRN-like bounded note).
- **At the prices the calls actually cost, it loses to the bounded note**: +196 [+150, +241].
  A real judge call costs 4.7 task rollouts (assumed 0.1) and a proposal 6.3 (assumed 1). Its
  prior expects small, often harmful rewrites, but 96% of the real rewrites were better (by
  about 45 points of success), so its propose gate stops asking too early.
- That is the RSI Tax result on real outputs, and the clearest case yet for learning the
  proposer's quality online. The real judge was right 76% of the time (assumed 70%).

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
git clone https://github.com/hegdev13/cost-aware-skill-admission.git && cd cost-aware-skill-admission
uv sync
uv run pytest -q                                   # 54 tests, about 15 s
uv run python experiments/build_viewer.py          # build the offline HTML viewer
open results/viewer/viewer.html                    # the interactive trace
```

### Demo (for talks and newcomers)

`demo/demo.html` is a single offline page that explains the project to someone who has never
seen it: (1) a race of every policy over the same 1,000-task stream in six cost regimes, with a
cost breakdown; (2) real decision records from the controller's log in plain English, with the
hidden truth revealed on demand; (3) "be the controller": a JavaScript port of the controller
where you move prices and evidence and watch the decision change. The page checks its port
against scores computed by the Python controller and shows the result.

```sh
open demo/demo.html                                # prebuilt, 50 seeds
uv run python demo/build_demo.py                   # rebuild (about 3-5 min; --seeds 5 for a quick one)
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
| `docs/06_replay_table_spec.md` | the real-LLM replay table, task suite, models, rate cap, and what the build changed (section 11) |
| `docs/07_results_report.md` | the results, the regime map, the ablations, the limitations |
| `src/ee_update/config.py` | every parameter and the ten named regimes |
| `src/ee_update/sim.py` | the simulator and the decision loop |
| `src/ee_update/policies.py` | baselines, oracle, belief controller, ablations |
| `src/ee_update/trace.py` | the append-only decision log |
| `src/ee_update/tui.py` | the terminal viewer |
| `experiments/` | grid, sweep, summaries, figures, viewer builder, claim verifier, artifact publisher |
| `src/ee_update/tasks.py` | the real-LLM task suite: 5 families with hidden conventions, exact-match verifiers |
| `src/ee_update/llm.py` | a cached Gemini client (real token counts; key from the environment) |
| `src/ee_update/replay.py` | replays any policy on the recorded table, charging assumed or measured prices |
| `data/replay/` | the recorded real-LLM table (skills, outcomes, judge verdicts, metadata) |
| `experiments/build_replay_table.py` | builds that table (needs `GEMINI_API_KEY`; cached and resumable) |
| `tests/` | 54 tests: simulator invariants, calibration, accounting, trace contract, viewers, replay, task suite |
| `figs/` | the six publication figures (PDF and SVG) |
| `results/` | generated CSVs, traces and the HTML viewer (git-ignored; rebuild with the scripts above) |

## Reproducing the numbers

```sh
uv run python experiments/run_grid.py --seeds 50     # 10 regimes x 17 policies x 50 seeds, about 30 min
uv run python experiments/run_sweep.py --seeds 20    # price x proposer-quality sweep, about 10 min
uv run python experiments/summarize.py               # writes results/tables.md
uv run python experiments/make_figures.py            # writes figs/fig1..fig6
uv run python experiments/run_replay.py              # replays every policy on the recorded real-LLM table, about 2 min
uv run python experiments/verify_claims.py           # re-checks every number quoted in the docs
uv run python experiments/publish_artifacts.py       # copies report, figures, data, viewer to the artifacts folder
```

`verify_claims.py` is the reproducible half of the verification log in `journal.md`: it
re-derives the headline table, the paired regime comparison, the ablation numbers, the sweep
claims and the quoted decision record from the raw CSVs and prints PASS/FAIL for each.

## Next steps

1. **Learn the proposer online.** The fixed prior is the controller's one real weakness in
   both directions: with a weak proposer it over-asks (loses to "never update"), and on the
   real table, where rewrites are large and almost always good, it under-asks (loses to the
   bounded note at measured prices).
2. **Scale risk with the stakes** to turn the costly-failure tie into a win without breaking
   the base regime.
3. **A bigger real-LLM replay**: harder families where rewrites can hurt, a skill tree
   (rewrites of the current incumbent), one call per task, more models, and the trace
   viewers on a real run.
4. Heavier-tailed candidate quality, and the held-out-regime check of the controller's
   hyperparameters.
