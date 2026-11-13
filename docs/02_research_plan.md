# Research plan: a cost-aware explore/exploit controller for agent skill updates

Status: plan only. No code has been run and no compute spent.

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
| P1 | Simulator + baselines + metrics + tests | CPU, minutes | Baselines reproduce expected orderings (accept-all worst under high-degradation, static best under no-signal) |
| P2 | Proposed controllers + ablations, 30+ seeds, regime sweep | CPU, minutes to hours | Clear map of where each policy wins, with confidence intervals |
| P3 | Optional theory: regret bound for the simplified setting | none | Bound or an honest statement that none was found |
| P4 | Real-agent validation: small task stream (ALFWorld-style), frozen small model, 3 to 5 policies | GPU or API; needs your approval and budget | Qualitative agreement with simulator ordering |
| P5 | Write-up | none | Paper draft + interactive figures |

**Kill criterion:** if after P2 the SRN-style baseline is within noise of our controller in every plausible regime, the contribution becomes a *negative/clarifying result* ("when does smart updating pay off?") and we reframe rather than force a win.

## 6. Risks
- Simulator realism (parameters come from reported statistics of different systems; mixing sources may not be coherent).
- OMCE/FGSE extensions could pre-empt the idea; mitigate with the cost-regime analysis and the real-agent check.
- Hidden-value assumption: real skills have context-dependent value, not one scalar. Add a context-dependent variant in P2.
- Overfitting controller hyperparameters to the simulator; keep a held-out regime set.

## 7. Immediate next steps (no compute)
1. Second read of AtomSearch cost section and BEAM's Beta head to settle the posterior family.
2. Decide the cost units (tokens only, or tokens plus a failure price) and the default price ratio.
3. Write the simulator spec (P0) as `docs/04_simulator_spec.md` before any code.
