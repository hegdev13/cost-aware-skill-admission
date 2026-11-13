# Research plan: a cost-aware explore/exploit controller for agent skill updates

Status: phases P1, P2 and P2b are complete; the results are in `docs/07_results_report.md`.
P3-P4 (the real-LLM replay table) have not started.

## 1. What is happening now vs what we want

| | **Now (field, Nov 2026)** | **Target (this project)** |
|---|---|---|
| Deciding if a new skill/lesson is kept | Fixed rule: accept-all, greedy pairwise judge (EvoGuard), probe gate with preset thresholds (FGSE), promote after m=2 episodes at precision 0.67 (StepLearn) | A belief over the candidate's value on *unseen* tasks (mean + uncertainty), updated as evidence arrives, and an action chosen from that belief |
| What a "test" costs | Treated as a hyperparameter (e.g. 8/16/16 probes in FGSE; whole validation sets in AtomSearch) | Test cost is part of the objective. Choices: free-ish live probation vs paid verification |
| Exploration | In retrieval (MEMOIR, BEAM) or in search trees (AtomSearch, UCT 0.05) | In **admission**: deliberately trying an uncertain candidate live to learn its value, when the expected gain justifies the risk |
| Whether to propose an update | Always (SRN, most methods) or agent-led text policy (Skill∞) | Explicit value-of-information rule: skip when the expected gain is below the cost of proposing and testing |
| Bandit credit | OMCE: reward = immediate utility of current query, no cost term seen | Reward = cost-denominated, with delayed credit to the update that created the skill |
| How methods are compared | Final success rate; RSI Tax adds total tokens | **Cost-regret against an oracle**, plus the accuracy/tokens frontier, across cost regimes |
| Strong simple baseline | SRN matches costly methods at 65.6–71.0% fewer tokens | We must match SRN's cost *and* beat it where candidates are risky |

One-sentence target: **a decision rule that, for every candidate update, picks skip / accept / live-probation / verify to minimize failures plus tokens, and a clear map of when it beats simple baselines.**

## 2. Research questions
- **RQ1.** In a controlled stream, does a belief-based controller achieve lower cost-regret than fixed-threshold rules (StepLearn-style, FGSE-style, EvoGuard-style greedy, accept-all, SRN-style bounded note)?
- **RQ2.** Does adding "whether to propose at all" (value-of-information) reduce total cost versus always proposing?
- **RQ3.** Which cost regimes (verification cost, failure cost, candidate quality spread, task non-stationarity) decide the winner? Where does the simple baseline win?
- **RQ4.** Do simulator conclusions hold in a small real-agent stream (validation only)?

## 3. Formal setup (to be refined in Phase 0)
- **Stream:** tasks t = 1..T from K task families; optional drift (a family's best skill changes at a change point).
- **State:** incumbent skill per family with value v (unknown to the agent); pool of candidates.
- **Candidate generation:** after a task, a proposer may emit a candidate with hidden true value drawn from a distribution calibrated to EvoGuard's rewrite statistics (about 42% worse, 8% tie, 50% better than parent; tails of ±10%).
- **Observations:** noisy success outcomes from live use; noisy verdicts from paid verification with a configurable judge accuracy and cost.
- **Controller actions per candidate:** SKIP, ACCEPT, PROBATE (route a fraction of matching tasks to the candidate and update its posterior), VERIFY (pay k validation rollouts, then accept/reject). Before the candidate exists: PROPOSE or NOT-PROPOSE.
- **Cost:** C_total = c_fail × (task failures) + c_tok × (task tokens + proposal tokens + probation overhead + verification tokens). All components logged separately (mirrors RSI Tax decomposition: verify / explore / write).
- **Metric:** cumulative cost-regret against an oracle that knows the true values and always picks the best skill without paying for testing. Secondary: final success, total tokens, harmful-commit rate, wasted-update share, steps to reach the best skill.

## 4. Methods to compare

**Baselines** (each mapped to a source):
1. Never update (static).
2. Accept-all.
3. SRN-style: one bounded note, no candidate search (RSI Tax).
4. Greedy pairwise judge with configurable accuracy (EvoGuard-style).
5. Threshold promotion: m=2 supporting episodes, precision 0.67 (StepLearn-style).
6. Probe gate with confidence bounds and fixed probe counts (FGSE-style, 8/16/16 and a sweep).
7. Always-verify: full validation before every accept (AtomSearch-style cost).

**Proposed controllers** (start simple, add only if needed):
- C1. Bayesian posterior per candidate (Beta or Gaussian) with a cost-aware thresholded decision: expected net gain vs cost of each action.
- C2. Thompson-sampling probation: candidate competes with the incumbent as a bandit arm; verification triggered only when posterior uncertainty and stakes are high.
- C3. C1 plus value-of-information gate on proposing.
- Ablations: no uncertainty (point estimates), no VoI gate, no delayed credit, no cost term.

## 5. Phases and gates (stop and re-plan at each gate)

| Phase | Work | Compute | Exit criterion |
|---|---|---|---|
| P0 | Fix formal model, calibrate simulator parameters from published numbers, pre-register metrics and seeds | none | Spec reviewed by you |
| P1 | Simulator + baselines + metrics + tests | CPU, minutes | **done**: 25 tests; calibration, accounting, oracle dominance and ordering checks pass |
| P2 | Proposed controllers + ablations, 30+ seeds, regime sweep | CPU, minutes to hours | **done**: 50 seeds x 10 regimes x 17 policies; the controller beats the best fixed rule in 7 regimes, ties in 1, loses in 1 |
| P2b | Decision-trace logger + HTML and TUI viewers on simulator runs (`docs/05_decision_trace_and_ui_spec.md`) | CPU, minutes | **done**: 29 tests including the HTML/TUI contract |
| P3 | Real-LLM replay table: pilot (~150 calls) then full build (~8,400 calls) with NVIDIA-hosted open models (`docs/06_replay_table_spec.md`) | API, free-tier assumed; capped at 12,000 calls and 20M tokens | Pilot criteria pass; table complete; real Δ distribution compared with the simulator |
| P4 | Replay all policies on the real table, regret vs oracle, same viewers on real traces; short live-mode demo | CPU for replay; small API use for the live demo | Policy ordering on real outcomes reported next to the simulator ordering |
| P5 | Optional: theory; optional multi-step (ALFWorld-style) stage; write-up | varies | Paper draft + interactive figures |

**Kill criterion:** if after P2 the SRN-style baseline is within noise of our controller in every plausible regime, the contribution becomes a *negative/clarifying result* ("when does smart updating pay off?") and we reframe rather than force a win.

## 6. Risks
- Simulator realism (parameters come from reported statistics of different systems; mixing sources may not be coherent).
- OMCE/FGSE extensions could pre-empt the idea; mitigate with the cost-regime analysis and the real-agent check.
- Hidden-value assumption: real skills have context-dependent value, not one scalar. Add a context-dependent variant in P2.
- Overfitting controller hyperparameters to the simulator; keep a held-out regime set.

## 7. Immediate next steps
1. **Learn the proposer's quality online.** The regime map shows the controller loses to
   "never update" when the proposer is weak (q = 0.25): its prior is fixed at the
   EvoGuard calibration, so it keeps buying candidates a calibrated agent would ignore.
   Updating the population prior from the candidates already tried is the natural fix and
   turns the rule into something that actually learns.
2. **Scale risk with the stakes.** The controller is risk-neutral and loses in the
   costly-failure regime (c_fail = 20). A margin that grows with c_fail, or a value-at-risk
   term, should recover it without breaking the base regime (a fixed 1-sigma margin destroys
   the rule, because the rewrite effect sizes are smaller than the prior's uncertainty).
3. Then the real-LLM replay stage (`docs/06_replay_table_spec.md`), which needs the API key.
