# Literature survey: explore/exploit inside the learning-update rule of LLM agents

Snapshot date: 2026-11-13. Scope: frozen-weight LLM agents that improve across tasks by editing external state (skills, memory, notes). The question is how the *update rule* decides what to try, what to keep, and what to spend.

**Evidence levels used below**
- **[read]**: I read the method, algorithm or appendix text of the paper.
- **[abstract]**: I saw only the abstract or search snippet. Do not cite these for claim-level support yet.

---

## 1. The shared problem

Single-episode rewrites are unreliable. [EvoGuard](https://www.alphaxiv.org/pdf/2611.06444) [read] measured 1,137 single-trajectory skill rewrites on ALFWorld: 42.2% scored below their parent skill, 7.7% tied, 50.0% improved. A stronger RL-trained generator still produced 40.0% degrading rewrites (900 rewrites). So the on-average effect of "accept every rewrite" is near zero, with a wide spread. Every paper below is a different answer to "which rewrites do we keep, and how much do we pay to decide?"

## 2. Four families of existing answers

### 2.1 Offline search over skill edits (expensive, no uncertainty model)
- **[AtomSearch](https://www.alphaxiv.org/pdf/2610.19951)** [read]: tree search over skill states with UCT (exploration constant 0.05), contrastive atomic edits, short lookahead, and an outcome memory of repair/regression counts. Accepts edits using a validation set. Cost: 923 to 17,200 item evaluations per run, 20 to 30 search attempts. Exploration here is search-tree exploration, not uncertainty about a skill's generalization.
- **[Skill∞](https://www.alphaxiv.org/pdf/2610.35434)** [read]: a meta-agent optimizes task skills and also rewrites its own optimization procedure; it allocates the search budget itself. 60.7% on 83 SkillsBench tasks vs 56.4% human-curated and 54.5% SkillEvolver. The budget allocation is agent-led text policy, not a stated decision rule.
- [SkillVine](https://www.alphaxiv.org/pdf/2609.32731), [SkillEvoReg](https://www.alphaxiv.org/pdf/2609.30861), [Skill-V](https://www.alphaxiv.org/pdf/2610.11781), [EVISKILL](https://www.alphaxiv.org/pdf/2610.05030), [Selm](https://www.alphaxiv.org/pdf/2610.24630), [SkillForge](https://www.alphaxiv.org/pdf/2610.09832) [abstract]: branching search, overfitting regularization, verifiable libraries, replay-grounded evolution. Not read.
- Survey: [Dynamic Agent Skills](https://www.alphaxiv.org/pdf/2607.10113) [abstract].

### 2.2 Training-time lifecycle managers (principled, but need RL rollouts)
- **[TrackRL](https://www.alphaxiv.org/pdf/2610.25126)** [read]: turns each GRPO group into a randomized experiment (skills injected independently with Bernoulli assignment). Tracks per-skill utility u_s with a standard error and a model-mastery estimate. A UCB-style term is used when *selecting which skills to offer*. Removal rule: after at least E_min evidence, remove if u_s + λ·SE(u_s) < 0. **Admission has no E/E**: new candidates come from reflecting on the lowest-scoring groups and are admitted if not a near-duplicate (similarity below δ_dup). Requires policy-training rollouts.
- **[Uni-Mem](https://www.alphaxiv.org/pdf/2610.29671)** [read]: one RL-trained memory manager for construction and retrieval (Stage-Decoupled GRPO, event-wise credit). Evaluated on LoCoMo and LongMemEval. No explicit E/E or cost term in what I read.
- [Before the Next Rollout](https://www.alphaxiv.org/pdf/2610.39358) (skill utility shifts after policy updates) [abstract].

### 2.3 Live gates on one stream (fast or safe, but fixed rules)
- **[EvoGuard](https://www.alphaxiv.org/pdf/2611.06444)** [read]: a Qwen3-4B LoRA selector compares old skill vs rewrite in one forward pass (24 ms). Trained on 24,000 pairs with at least a 10% success gap, using a private 100-task test set per task family. Gains vs accept-all: +11.0% ALFWorld, +5.9% ScienceWorld, +2.7% LiveCodeBench. **Purely greedy**: one comparison, accept the winner, never revisit, no uncertainty output.
- **[FGSE](https://www.alphaxiv.org/pdf/2611.14326)** [read]: each candidate lesson is a structured hypothesis (precondition, behavior, effect, verifier), tested in a temporary state on target-transfer, falsification and regression probes, then commit / refine / reject. Gate: lower confidence bound on gain ≥ η = 0.03, upper bound on falsification rate ≤ δ = 0.05, upper bound on regression damage ≤ β = 0.01. Defaults: 8 target + 16 falsification + 16 regression probes per candidate, 3 candidates per episode. In one run it committed 104 of 380 candidates; the rejected ones above the gain threshold had a 22.82% falsification rate vs 1.28% for committed. Has confidence bounds, but the probe count and thresholds are fixed hyperparameters tuned on dev tasks, and probe cost is not traded against expected benefit. The paper does include a probe-budget sweep ablation.
- **[StepLearn](https://www.alphaxiv.org/pdf/2609.35911)** [read]: use a hypothesis immediately for the next step, promote it to persistent knowledge only after prospective validation: at least m = 2 supporting episodes and precision ρ = 0.67. Beats EvoTest by 2.2 to 12.7 points on WebArena-Lite and ALFWorld. A fixed promotion threshold, no cost model.
- **[ENGRAM](https://www.alphaxiv.org/pdf/2611.09472)** [read]: verified release gate: records leave retrieval only if the updated weights match the retrieval-supported base on reserved probes. A deterministic gate on a different decision (when to delete).
- **[MEMOIR](https://www.alphaxiv.org/pdf/2611.07631)** [read]: after each query, retrieval priority is revised by attribution; a UCB-style bonus encourages retrieving under-exposed memories; merges are certified by a 3-judge check (0 observed losses in 100 audited merges, one-sided 95% upper bound 2.95%). E/E is on **retrieval**, not on writing.

### 2.4 Bandit or Bayesian components on the memory side
- **[BEAM](https://www.alphaxiv.org/pdf/2610.25119)** [read]: memory value as a context-conditioned Beta belief (α, β) predicted by a LoRA-adapted estimator; retrieval uses Thompson sampling combined with similarity (six candidates retrieved, up to three injected). This is the closest existing "uncertainty over memory value", but it governs **retrieval of existing memories**, not whether a new candidate is admitted or tested.
- **[OMCE](https://www.alphaxiv.org/pdf/2611.01747)** [read]: a contextual bandit (Thompson sampling or LinUCB, forgetting factor 0.98) chooses among DIRECT / EPISODE / GUIDE / DISTILL per query; DISTILL is the "when to consolidate" decision. Uses 5.7 to 62.5 times fewer tokens per task than ACE. Important detail: the bandit reward r_t ∈ [0, 1] is the **immediate utility of the current query**. So a distillation is credited only by the very next query, not by its delayed, cross-task value, and I found no explicit cost penalty in the reward.
- Earlier-search items from a previous assistant reply, not verified by me [abstract]: SkillBoost (constrained repair search), APEX (strategy map with fork selection), MemCon, Janus, DAEDALUS.

## 3. Cost accounting: the one paper that sets the bar

**[The RSI Tax](https://www.alphaxiv.org/pdf/2611.13524)** [read] counts the tokens a self-improvement process spends (even for rejected updates) alongside task-execution tokens.
- A bounded-note baseline (SRN: revise one note from completed failures, no candidate search) matched or beat Meta-Harness, EvoSkill and RRSI in accuracy while using 65.6 to 71.0% fewer total tokens.
- Verification was 74.8 to 93.6% of aggregate update cost for those three methods.
- A large share of update cost went to artifacts never used later: of each method's own update cost, 63.57% (ReasoningBank), 52.78% (Meta-Harness), 37.91% (EvoSkill) and 23.70% (RRSI) had no downstream impact in the evaluated streams.
- The authors call for objectives that jointly optimize update cost and task cost. They do not propose a decision rule.

Consequence for us: any controller must be judged on **total cost**, and must beat or match the simple bounded-note baseline.

## 4. What is covered vs not (verified against the papers above)

| Question | Covered by | Not covered (as far as I read) |
|---|---|---|
| Does a candidate generalize? | EvoGuard (pairwise, greedy), FGSE (probes + confidence bounds), StepLearn (promotion threshold), TrackRL (RCT inside RL) | A *sequential* belief that is updated by cheap live trials and by paid verification, then acted on |
| Bayesian value with uncertainty | BEAM (retrieval), TrackRL (SE on skills, needs RL), FGSE (bounds) | Used for the **admit / try / verify** decision on a new candidate |
| Bandit controlling memory operations | OMCE (use / form / distill), MEMOIR (retrieval bonus) | Reward with cost and with **delayed** credit for an update |
| Whether to propose an update at all | SRN in RSI Tax (always one note), Skill∞ (agent-led budget) | A value-of-information rule: propose only if expected gain exceeds cost |
| Cost-aware evaluation | RSI Tax (accuracy vs tokens, cost decomposition) | Regret against an oracle that knows true skill values; policy comparison across cost regimes |

## 5. The gap we target

1. **Admission as a sequential decision with explicit costs.** After a candidate appears, choose among skip / accept / live probation / paid verification, using a belief about its value over unseen tasks. Fixed thresholds (StepLearn, FGSE) and greedy pairwise (EvoGuard) are the comparison points.
2. **Value-of-information for writing.** Decide whether proposing or editing is worth its tokens given recent failure rate and how uncertain the current skill is. RSI Tax shows the waste; OMCE shows a bandit can control consolidation, but with immediate reward.
3. **Evaluation with a known oracle.** Real systems never know a skill's true value, so regret cannot be measured. A simulator calibrated to published rewrite statistics gives cost-denominated regret, which is not reported in the papers above.

## 6. Risks to the novelty claim (be honest early)
- OMCE or FGSE could be extended to our setting with small changes; a reviewer may see ours as incremental unless the cost-regime analysis is strong.
- RSI Tax's simple baseline may already be near-optimal in many regimes; the honest result may be "complex controllers only win when verification is cheap or candidates are high-variance".
- Simulator results do not prove real-agent gains; a small real-agent validation is required before any strong claim.
- Several neighbors above were not read in full (see [abstract] tags); do a second pass before writing related work.
