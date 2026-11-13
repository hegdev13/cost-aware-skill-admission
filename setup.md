# Setup and usage

Everything here was run on a fresh clone of branch `ee-update-rule` on 2026-11-13 (macOS, Apple silicon):
`uv sync`, the test suite (31 pass), building the viewer, and the terminal viewer's usage message.
Nothing in this project needs a GPU, an API key, or a network connection, except the real-LLM stage, which is
specified but not built (section 9).

## 1. What you need

| Need | Why | Check |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | installs Python 3.10+ and the dependencies | `uv --version` |
| git | clone the repository | `git --version` |
| A web browser | to open `viewer.html` | any modern browser |
| A terminal | for the terminal viewer | |
| Google Chrome (optional) | one test runs the HTML viewer in headless Chrome; it skips itself if Chrome is missing. Set `CHROME_BIN` if Chrome is not in `/Applications/Google Chrome.app` | |

Do not use the system `python3` directly (it may be old and lacks numpy). Always go through `uv run`.

## 2. Install

```sh
git clone --branch ee-update-rule <repo-url> ee-update && cd ee-update
uv sync
uv run python -c "import ee_update; print('ok')"
```

If the last line says `No module named 'ee_update'`, repair the editable install:

```sh
uv sync --reinstall-package ee-update
```

## 3. Check that everything works

```sh
uv run pytest -q          # expect: 31 passed, about 10 s
```

The tests cover: determinism, the candidate-quality calibration against the published rewrite statistics, the cost
accounting identity for every policy, the oracle never losing when proposals are free, static beating accept-all with
no good candidates (and the reverse), policies not seeing hidden values, the trace contract, the terminal viewer
rendering headlessly, and the HTML viewer running in a real browser.

## 4. Build and open the HTML viewer

`results/` is git-ignored, so a fresh clone has no viewer yet. Build it (about 2 seconds, deterministic):

```sh
uv run python experiments/build_viewer.py                  # seed 3, base regime
uv run python experiments/build_viewer.py --seed 7 --regime D_weak_proposer   # any seed and regime
open results/viewer/viewer.html                            # macOS; on Linux use xdg-open
```

This writes `results/viewer/viewer.html` (about 3 MB, one self-contained file, no server, no network) and one
`<policy>.trace.jsonl` plus `<policy>.truth.jsonl` per policy. Regimes you can pass:
`base, A_cheap_verify, B_costly_verify, C_costly_fail, D_weak_proposer, E_gen_gap, F_drift, G_dear_judge,
H_noisy_judge, J_no_judge`.

### How to read the viewer

**Top bar**
- **Policy**: which policy's run to show (`c3_voi_propose` is "ours"; also `c1_voi`, `greedy_judge`,
  `accept_all`, `threshold_promote`, `static`).
- **Compare**: add a second policy's task strip under the first, on the same seed and the same candidates.
- **Seed / Regime**: what the run was built with.
- **play / pause**: animates the scrubber.
- **truth layer**: off by default. Turn it on to see each candidate's true value and true advantage over its
  parent, next to the decision that was made about it.
- **explore only**: hides decisions that did not buy information.
- **Scrubber**: move through the stream; everything shown is "as of task t".

**Left panel**
- **Task stream, per family**: one lane per task family. Each tick is a task: blue = success, orange = failure,
  green = served by a candidate on a live trial. Marks: triangle = candidate created, black circle = skill
  accepted, grey square = candidate rejected. Hover any mark for the task, the skill and the rule that routed it.
- **Cost ledger**: a stacked bar and a table of where the cost went (failures, task rollouts, proposals,
  verification, judge) up to the scrubber, plus the reference cost (the cheaper of never-updating and the
  free-information oracle) and the regret so far.

**Right panel (tabs)**
- **decisions**: one card per decision, oldest first. Each card shows the candidate, the action
  (`SKIP / ACCEPT / PROBATE / VERIFY / JUDGE`, or `PROPOSE / HOLD` for the proposal decision), an
  EXPLORE or EXPLOIT chip, the one-sentence reason, the rule id, the belief and inputs
  (`mu_D` = expected advantage over the incumbent, `s_D` = spread, `p_better`, horizon, ...), the greedy action
  (what it would do without buying information), and a bar for the net value of **every** action it considered.
- **belief**: the controller's expected advantage over time with its spread, coloured by the action taken.
- **lineage**: every candidate, its parent, whether it was accepted or rejected, how many tasks it served, and the
  rule that decided (plus true value when the truth layer is on).
- **cost table**: totals for tasks, failures, success rate, each cost component, and the number of decisions that
  were EXPLORE.

Note: `s_D` shown on cards includes uncertainty about the incumbent's own value, so it can be wider than the
candidate's own posterior (see `journal.md` section 9).

## 5. The terminal viewer

```sh
uv run python -m ee_update.tui results/viewer/c3_voi_propose.trace.jsonl
```

It reads the same log as the HTML page. Keys: `space` play or pause, `n` next decision, `b` previous decision,
`p` switch policy (all `*.trace.jsonl` in the same folder), `t` toggle the truth layer, `e` explore-only,
`/` search a skill id, `q` quit. Panels: the task strip, the belief bar for the active candidate, a scrolling
decision log with the reason text, the ledger, and the lineage list. Build the viewer first (section 4) so the trace
files exist. A wide terminal works best.

## 6. Reproduce the results

```sh
uv run python experiments/run_grid.py --seeds 50     # 10 regimes x 17 policies x 50 seeds: about 30 minutes
uv run python experiments/run_sweep.py --seeds 20    # price x proposer-quality sweep: about 10 minutes
uv run python experiments/summarize.py               # writes results/tables.md
uv run python experiments/make_figures.py            # writes figs/fig1..fig6 as PDF and SVG
uv run python experiments/verify_claims.py           # re-checks every number quoted in the docs
uv run python experiments/tune_margin.py             # the acceptance-margin check (about 10 minutes)
```

Quick smoke test (a couple of seconds, writes to a scratch folder, does not touch `results/`):

```sh
uv run python experiments/run_grid.py --seeds 2 --regimes base \
    --policies static,greedy_judge,c3_voi_propose --out /tmp/grid_smoke
```

Outputs: `results/grid.csv` (one row per regime, policy, seed), `results/curves.csv` (cumulative cost curves, about 90 MB),
`results/sweep.csv`, `results/tuning.csv` (the margin check), `results/*_meta.json` (what was run and how long it took).

`verify_claims.py` prints one PASS/FAIL line per claim quoted in `docs/07_results_report.md` and
`journal.md` (41 checks: the headline table, the paired regime comparison, the ablation numbers, the
sweep claims, the quoted decision record, the recorded run metadata) and exits non-zero if any fails.
Run it after regenerating the data.

`experiments/publish_artifacts.py` copies the report, figures, data and viewer into the OpenResearch artifacts folder.
That path is hard-coded at the top of the script; edit it before using it elsewhere.

## 7. Where things are

| Path | What |
|---|---|
| `src/ee_update/config.py` | all parameters and the ten named regimes |
| `src/ee_update/sim.py` | the simulator and the decision loop |
| `src/ee_update/policies.py` | baselines, oracle, belief controller and ablations |
| `src/ee_update/trace.py` | the decision log writer and reader |
| `src/ee_update/tui.py` | the terminal viewer |
| `experiments/` | grid, sweep, summaries, figures, viewer builder, artifact publisher |
| `tests/` | 31 tests |
| `experiments/verify_claims.py` | re-checks every number quoted in the docs against the raw CSVs |
| `experiments/tune_margin.py` | the acceptance-margin check over the four development regimes |
| `docs/` | literature survey, plan, specs, results report |
| `figs/` | figures (PDF and SVG) |
| `results/` | generated data and viewer (git-ignored) |

## 8. Trying your own policy or regime

- **A new regime**: add an entry to `REGIMES` in `src/ee_update/config.py` (overrides of `Config` fields), then pass its
  name to `build_viewer.py --regime` or `run_grid.py --regimes`.
- **A new policy**: subclass `Policy` in `src/ee_update/policies.py`, implement `propose(view)` and `decide(view)`
  returning a `Decision` (action, rule id, inputs, scores, reason), and register it in `make_policy`. A policy only
  receives observable data (`View`); hidden values are never passed in.
- Every `Decision` you return appears in the viewer exactly as written, so give it a good `reason`.

## 9. The real-LLM stage (not built)

`docs/06_replay_table_spec.md` describes a replay table built with open models served on an NVIDIA API key.
It needs `NVIDIA_API_KEY` set in your shell (set it yourself; do not paste it into chat, a file or a commit).
To see which models the key lists (one metadata request, no inference):

```sh
curl -s -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models | head -c 600
```

The rate cap in that spec (30 requests per minute, concurrency 4, 12,000 calls, 20M tokens) is a conservative
choice by the project, not a documented limit of the service.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ee_update'` | `uv sync --reinstall-package ee-update` |
| `ModuleNotFoundError: No module named 'numpy'` | you ran the system `python3`; use `uv run python ...` |
| `viewer.html` does not exist | run `uv run python experiments/build_viewer.py` (results are git-ignored) |
| Viewer page is blank | you have an old copy built before the 2026-11-13 fix; rebuild it |
| Browser test skipped | Chrome not found; set `CHROME_BIN=/path/to/chrome` or ignore the skip |
| `run_grid.py` is slow | that is expected (about 30 minutes for the full grid); use `--seeds` and `--regimes` to shrink it |
| `verify_claims.py` fails | a number in the docs no longer matches `results/`; either the data was regenerated with a different code version, or a claim needs updating - do not edit the number without checking the CSV |
