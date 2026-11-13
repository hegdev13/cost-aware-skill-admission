# ai-exploration-exploitation

Explore/exploit inside the learning-update rule of LLM agents: when should an agent keep, test, verify or skip a candidate skill/memory update, given that testing costs tokens and failures cost accuracy.

Status: the simulator, the fixed baselines and the cost-aware belief controller are
built, tested and measured (docs/07_results_report.md). The real-LLM replay stage
(docs/06_replay_table_spec.md) has not started; it needs an API key.

## Read in this order
1. `docs/01_literature_survey.md`: what exists, what is covered, the gap.
2. `docs/02_research_plan.md`: now vs target, research questions, setup, phases, kill criterion.
3. `docs/03_outputs_and_presentation.md`: figures, tables, formats, possible outcomes.
4. `docs/04_simulator_spec.md`: simulator design, calibration, baselines, tests.
5. `docs/05_decision_trace_and_ui_spec.md`: event log, explore/exploit labeling, HTML and TUI viewers.
6. `docs/06_replay_table_spec.md`: real-LLM replay table, task suite, models, rate cap, pilot.

## Layout
- `docs/`: survey, plan, specs
- `src/ee_update/`: simulator, policies, decision trace, terminal viewer
- `configs/`: experiment configs (YAML, seeds listed)
- `experiments/`: run scripts
- `results/`: raw CSVs, traces, the HTML viewer (git-ignored)
- `figs/`: publication figures (PDF + SVG)
- `tests/`: unit tests
