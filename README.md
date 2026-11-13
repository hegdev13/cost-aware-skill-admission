# ai-exploration-exploitation

Explore/exploit inside the learning-update rule of LLM agents: when should an agent keep, test, verify or skip a candidate skill/memory update, given that testing costs tokens and failures cost accuracy.

Status: planning only. No experiments have been run.

## Read in this order
1. `docs/01_literature_survey.md`: what exists, what is covered, the gap.
2. `docs/02_research_plan.md`: now vs target, research questions, setup, phases, kill criterion.
3. `docs/03_outputs_and_presentation.md`: figures, tables, formats, possible outcomes.

## Layout
- `docs/`: survey, plan, specs
- `src/ee_update/`: simulator, policies, metrics (empty until Phase 1)
- `configs/`: experiment configs (YAML, seeds listed)
- `experiments/`: run scripts
- `results/`: raw CSVs (git-ignored)
- `figures/`: generated figures (git-ignored)
- `tests/`: unit tests
