# Decision trace and UI specification (HTML + TUI)

Status: specification only. Nothing built or run.

## 1. Principle
**One trace, many views.** Every simulator run and every real-LLM replay writes the same append-only event log (JSONL). The HTML page and the terminal UI are only renderers of that log. Nothing is shown that is not in the log, so what you show your advisor is exactly what the system did.

## 2. What "transparent" means here
For every skill, the viewer must answer, with no guessing:
- **Where did it come from?** Created at step t, by which proposer call, from which failed tasks, as an edit of which parent. Full text diff vs parent.
- **Why was it used?** Which skill served each task and which routing rule chose it.
- **Why was it accepted, rejected or kept on trial?** The rule that fired, its inputs (posterior mean and spread, costs, thresholds), and the scores of the alternatives it did not pick.
- **What did it cost?** Failures and tokens charged to this decision, by component.
- **Was it right?** After the fact, the true value from the full table, shown in a separate "truth" layer so it can be hidden while the policy runs.

## 3. Event schema (one JSON object per line)
Common fields on every event: `run_id`, `seed`, `policy`, `regime`, `t` (task index), `seq` (event order), `family`, `ts_sim` (simulated time only; no wall-clock).

| `type` | Extra fields | Meaning |
|---|---|---|
| `TASK_SERVED` | `skill_id`, `routing_rule`, `outcome` (0/1), `tokens`, `arm` (`incumbent` or `candidate`) | A real task was run with this skill. This is the "skill invoked" record |
| `PROPOSE_DECISION` | `chosen` (yes/no), `rule_id`, `rule_inputs`, `expected_gain`, `cost` | Whether to ask the proposer for a candidate |
| `SKILL_CREATED` | `skill_id`, `parent_id`, `source_task_ids`, `proposer_call_id`, `text_diff`, `tokens` | A candidate came into existence |
| `ACTION_CHOSEN` | `candidate_id`, `action` (SKIP/ACCEPT/PROBATE/VERIFY), `mode` (EXPLORE/EXPLOIT), `rule_id`, `rule_inputs`, `scores` (score of every action considered), `reason` | The core decision record |
| `PROBATION_UPDATE` | `candidate_id`, `n_cand`, `wins_cand`, `n_inc`, `wins_inc`, `posterior` (mean, sd, P(better)) | Belief update after a trial result |
| `VERIFY_RESULT` | `candidate_id`, `k`, `successes`, `tokens`, `posterior_after` | Outcome of paid verification |
| `SKILL_ACCEPTED` | `skill_id`, `replaced_id`, `rule_id`, `rule_inputs` | Incumbent replaced ("updated") |
| `SKILL_REJECTED` | `skill_id`, `rule_id`, `rule_inputs` | Candidate discarded |
| `COST_CHARGED` | `component` (failure, task, proposal, probation, verify, judge), `amount`, `caused_by` (event `seq`) | Ledger entry, so totals can be audited |
| `TRUTH` (separate layer, `oracle_only: true`) | `skill_id`, `true_value`, `parent_true_value`, `was_good_decision` | Revealed post-hoc; never given to the policy |

`reason` is generated from a fixed template filled with `rule_inputs` (no LLM), for example: *"Candidate c7: P(better than incumbent) = 0.58, 4 trials, 1 more probation task costs ≈ 0.4 expected failures vs expected gain 1.9 → PROBATE (EXPLORE)."* Each `rule_id` is documented in a rules registry file so every message links to a written definition.

## 4. Explore vs exploit: exact labeling
- **EXPLOIT:** the action that maximizes the policy's current expected value without buying information. Using the incumbent, accepting a candidate whose posterior already clears the bar, skipping a clearly bad one.
- **EXPLORE:** any action chosen because of what it will teach, at a cost: probation (a share of tasks goes to an uncertain candidate), verification, or proposing a new candidate.
- Each `ACTION_CHOSEN` event carries both the greedy-best action and the chosen one. If they differ, it is EXPLORE and the log records the value of information the rule assigned. Baselines that have no belief (accept-all etc.) still emit events; their `mode` is derived from the same definition, so the viewer can show that, for example, accept-all never explores and never verifies.

## 5. HTML viewer
- **Single self-contained file** generated from a trace (no server, opens by double-click, works offline).
- **Layout:**
  1. Header: policy picker, regime, seed, play/pause, scrubber over t.
  2. Timeline lane per family: dots for tasks (success/fail), colored bars for which skill served; markers for creation, accept, reject.
  3. **Belief panel:** posterior band for the current candidate with the incumbent's estimate, and the decision threshold line.
  4. **Decision log:** every `ACTION_CHOSEN` as a card showing rule, inputs, alternatives and the `reason` sentence; EXPLORE/EXPLOIT chips.
  5. **Skill lineage tree:** nodes are skills, edges are edits; click shows the text diff, creation reason, tasks served, and (if the truth layer is on) true value.
  6. **Cost ledger:** running stacked bars by component, plus regret vs oracle.
  7. **Truth toggle:** off while stepping through; on to see which decisions were good.
- **Compare mode:** two policies on the same seed, aligned by t, so differences in decisions are visible side by side.

## 6. TUI viewer
- Python **Textual** app (a terminal UI library; not installed on this machine yet, to be added through uv in the project environment), reading the same JSONL.
- Panes: task stream strip, belief bar for the active candidate, scrolling decision log with the `reason` text, ledger, lineage list.
- Keys: `space` play/pause, `n` jump to next decision, `b` previous, `p` switch policy, `t` toggle truth layer, `e` filter to EXPLORE only, `/` search a skill id, `q` quit.
- **Live mode** (later): same app consuming events as a run produces them, with a banner showing the cumulative call count and budget left.

## 7. Contract tests (before any UI is trusted)
1. Every `COST_CHARGED` sum equals the run total.
2. Every `SKILL_ACCEPTED` and `SKILL_REJECTED` follows an `ACTION_CHOSEN` or `VERIFY_RESULT` for the same skill.
3. Every skill referenced in `TASK_SERVED` has a `SKILL_CREATED` (or is the empty seed).
4. `TRUTH` events never appear before the run ends in the policy-visible stream.
5. HTML and TUI render the same trace to the same decision count and the same final skill.
