# Simulator specification (Phase P0)

Status: specification only. Nothing is implemented or run. Every number marked **config** is a default we sweep or revisit; every number marked **derived** or **source** says where it comes from.

## 1. Purpose and non-goals
- **Purpose:** a cheap, fully controlled stream in which the true value of every skill is known to the simulator but hidden from the policy, so cost-regret against an oracle can be measured and policies compared under identical randomness.
- **Non-goals:** it does not simulate language, real agents, or real benchmarks. It tests *decision logic only*. Real-agent claims need Phase P4.

## 2. World model

### 2.1 Entities
- **Families** f = 1..K (**config** K = 6, matching ALFWorld's six task types). Each task belongs to one family, drawn uniformly (**config**; optional skewed variant).
- **Skill** s has a hidden scalar value v_s in [0, 1] = probability the agent succeeds on a task of its family when using s.
- **Incumbent** per family: the skill currently used. Starts as the empty skill with v_0 drawn from Beta(a0, b0) (**config** mean 0.5; sweep 0.3 to 0.7).
- **Candidate:** a proposed edit of the incumbent with hidden value v_c.

### 2.2 Candidate value (calibrated)
Parent value v_p; candidate value v_c = clip(v_p + Δ, 0, 1) with Δ from a three-part mixture.

**Source:** [EvoGuard](https://www.alphaxiv.org/pdf/2611.06444) reports, for 1,137 single-trajectory rewrites: 42.2% score below parent, 7.7% tie, 50.0% improve; and 7.7% lose at least 10 points, 11.3% gain at least 10 points. (Shares sum to 99.9% in the paper; we renormalize.)

| Component | Probability | Distribution of Δ | Derivation |
|---|---|---|---|
| Worse | 0.422 | −Exponential(scale 0.059) | **derived**: P(Δ ≤ −0.10 given worse) = 7.7/42.2 = 0.182; for an exponential this gives scale = 0.10 / ln(1/0.182) ≈ 0.059 |
| Tie | 0.077 | 0 | source |
| Better | 0.500 | +Exponential(scale 0.067) | **derived**: P(Δ ≥ 0.10 given better) = 11.3/50.0 = 0.226 → scale ≈ 0.10 / ln(1/0.226) ≈ 0.067 |

The exponential tail is our assumption; EvoGuard reports only the two tail masses. Sensitivity runs with a heavier (lognormal) tail are required. In expectation a rewrite is nearly neutral, as the paper states; that is the point of the benchmark.

**Proposer quality knob (config):** multiply the "better" probability by q and renormalize (q = 1 is the calibrated default; q < 1 models a weak proposer; EvoGuard notes a stronger generator still produced 40.0% degrading rewrites, so even a strong proposer keeps q below about 1.2).

### 2.3 Outcome noise
- Live use of skill s on a task: success ~ Bernoulli(v_s).
- Optional **drift** (**config**, off by default): each skill's value follows v_s(t+1) = clip(v_s(t) + N(0, σ_d²)) with σ_d in {0, 0.002, 0.01}.
- Optional **context variant V2** (Phase P2): each family has M sub-contexts; a candidate's value in sub-context m is v_c + shared + independent noise, with correlation ρ_ctx. A policy that tests on one sub-context overestimates generalization. This is the "unseen tasks" risk.

### 2.4 Observation channels available to a policy
1. **Live outcomes** of whatever skill was used on each real task (free, but a failure costs c_fail).
2. **Probation:** route a fraction q_p (**config** 0.2) of a family's tasks to the candidate while the incumbent serves the rest. Both arms' outcomes accumulate in a Beta posterior. Exploration cost appears naturally as extra failures if the candidate is worse.
3. **Rollout verification:** run k held-out rollouts of the candidate (**config** k in {5, 10, 20, 40}); each costs τ_roll tokens and returns Bernoulli(v_c + shift) where **shift ~ N(0, σ_shift²)** models a train/held-out generalization gap (**config** σ_shift = 0 default; sweep 0 to 0.1).
4. **Judge verdict** (EvoGuard-style): a cheap comparison of candidate vs incumbent that returns the truly better skill with probability a (**config** a = 0.7; sweep 0.55 to 0.9), at cost τ_judge (**config** 0.02 rollouts).

The policy never sees v_s, Δ, drift, or the shift.

## 3. Decisions
At each **update opportunity** (after a task, **config** every task or every n-th task):
1. **PROPOSE or NOT-PROPOSE.** Proposing costs τ_prop (**config** 1 rollout-equivalent; sweep 0.25, 1, 4). If proposed, a candidate arrives.
2. For each live candidate, one action:
   - **SKIP** (discard),
   - **ACCEPT** (replace incumbent immediately),
   - **PROBATE** (start or continue probation, then decide later),
   - **VERIFY(k or judge)** (pay for verification, then accept or reject on the result).
3. A policy may revisit a probated candidate at later opportunities.

Constraint: at most C_max candidates in flight per family (**config** 2).

## 4. Cost model
Unit: one task rollout = 1 token-unit (**config**; keeps results independent of real token counts).

C_total = c_fail × (number of failed real tasks) + c_tok × (task rollouts + proposal + verification + judge)

- **c_fail** in units of rollouts (**config** 5; sweep 1, 5, 20). This is the price of a bad outcome relative to compute.
- Probation adds no explicit tokens beyond the real tasks; its cost is the failures it risks.
- Logged separately per policy and seed: task failures, proposal, probation-induced failures, verification, judge, wasted (accepted or tested skills that were later replaced without ever serving more than w tasks; **config** w = 10). This mirrors the verify/explore/write decomposition in [The RSI Tax](https://www.alphaxiv.org/pdf/2611.13524).

Note: the RSI Tax paper found verification dominated update cost (74.8 to 93.6%) for three costly methods; the default grid must include a regime where verification is expensive relative to a single task.

## 5. Stream and horizon
- T = 1,000 real tasks (**config**; also 200 and 5,000 for horizon sensitivity).
- Seeds: 50 for reported results; 5 for development.
- **Common random numbers:** for each seed, pre-draw (a) the family sequence, (b) one uniform per task for outcome noise, (c) a table of candidate (Δ, uniform streams) indexed by opportunity number, (d) verification and judge noise tables. Every policy reads the same tables. Then paired differences between policies are low-variance and results are reproducible.

## 6. Oracles and regret
- **Oracle O1 (free information):** proposes at every opportunity (pays τ_prop), sees the true v_c, adopts if v_c > v_inc, pays no testing cost, and serves every task with the best known skill.
- **Static:** never proposes; keeps the empty skill.
- **Reference cost** = min(C(O1), C(Static)) on the same seed, so that skipping unprofitable proposals is not scored as beating the oracle.
- **Cost-regret** of policy π = C(π) − reference, cumulative over t. Report the full curve and the final value.

## 7. Policy interface
```
class Policy:
    def reset(self, config): ...
    def propose(self, history) -> bool
    def on_candidate(self, candidate_id, history) -> Action   # SKIP | ACCEPT | PROBATE | VERIFY_ROLLOUT(k) | VERIFY_JUDGE
    def on_result(self, event): ...                           # outcomes, verdicts
```
`history` contains only observable data: outcomes, verdicts, costs paid, counts. Hidden values live in the simulator and are never passed in. A test must assert this by running policies against a simulator whose hidden values are randomized after the fact.

## 8. Baselines to implement first (all fixed, no learning)
| Name | Rule | Source |
|---|---|---|
| Static | never propose | – |
| Accept-all | propose each time, accept | common baseline in [EvoGuard](https://www.alphaxiv.org/pdf/2611.06444) |
| SRN-like | propose after failures only, one slot, overwrite | [The RSI Tax](https://www.alphaxiv.org/pdf/2611.13524) |
| Greedy judge | VERIFY_JUDGE with accuracy a, accept the winner | EvoGuard |
| Threshold promote | PROBATE; accept when at least m = 2 supporting wins and win rate at least 0.67 | [StepLearn](https://www.alphaxiv.org/pdf/2609.35911) |
| Probe gate | VERIFY_ROLLOUT with fixed k, accept if lower confidence bound on gain ≥ η = 0.03 | [FGSE](https://www.alphaxiv.org/pdf/2611.14326) (η from its defaults; its probe counts are 8/16/16) |
| Always-verify | VERIFY_ROLLOUT(k large) before every accept | AtomSearch-style cost |

We copy parameters from the papers but our simulator differs; these are "-style" baselines, not reproductions, and the write-up must say so.

## 9. Proposed controllers (Phase P2, not yet specified in code)
- **C1:** per-candidate Beta posterior over (v_c − v_inc); choose the action with the largest expected net gain minus its cost, using a lookahead of H tasks (**config** H = remaining horizon or a fixed window).
- **C2:** Thompson-sampling probation, with verification triggered only when posterior uncertainty times stakes exceeds the verification price.
- **C3:** C1 plus an explicit value-of-information test for PROPOSE.
Ablations: point estimates only, no VoI gate, no delayed credit, no cost term.

## 10. Experiment grid
Named regimes (all defaults otherwise):
| Regime | What changes |
|---|---|
| A: cheap verification | k = 5, τ_judge small, c_fail = 5 |
| B: expensive verification | k = 40, c_fail = 5 |
| C: costly failures | c_fail = 20 |
| D: weak proposer | q = 0.5 |
| E: generalization gap | σ_shift = 0.08 |
| F: drift | σ_d = 0.01 |

Main grid: 6 regimes × 7 baselines (+ 3 controllers later) × 50 seeds. Heatmap sweep for F4: verification cost × proposer quality × c_fail on a coarser seed budget. The compute estimate is minutes on a laptop CPU; it is unmeasured.

## 11. Validation tests (written before trusting any result)
1. **Determinism:** same seed and config give identical results.
2. **Calibration check:** with 10^6 sampled Δ, the empirical shares (worse, tie, better) and tail masses match the table within Monte-Carlo error.
3. **Accounting identity:** the sum of logged cost components equals C_total.
4. **Oracle sanity:** O1 cost ≤ every policy's cost when τ_prop = 0 and c_fail > 0.
5. **Static ordering:** with q → 0 (no good candidates), Static beats Accept-all.
6. **Accept-all ordering:** with q large and σ_shift = 0, Accept-all beats Static.
7. **Information hiding:** policies cannot read hidden values (see §7).
8. **CRN check:** two policies that make the same decisions produce identical trajectories.

## 12. Known limitations (will be stated in the paper)
- A skill's value is a scalar per family; real skills are context-dependent (V2 partly addresses this).
- Candidate quality is independent across proposals; real proposers learn from earlier feedback.
- Cost units are abstract; mapping to real tokens needs the P4 check.
- The EvoGuard calibration comes from one benchmark (ALFWorld) and one generator.

## 13. Open questions for you
1. Cost units: keep "rollout-equivalents" (recommended) or convert to dollars later?
2. Default c_fail = 5 rollouts: is that a sensible stand-in for your intended setting?
3. Include context variant V2 in the first implementation, or add after the main results?
