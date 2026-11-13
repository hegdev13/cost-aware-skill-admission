"""Skill-update stream simulator (docs/04_simulator_spec.md).

Hidden values live only in Skill.v / World tables. Policies receive a View built from
observable data. The oracle policy is the only one handed a reference to the simulator.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .trace import Trace

NMAX = 300   # proposals per family available in the pre-drawn tables
VMAX = 300   # verification rollouts per candidate in the tables

_WORLD_CACHE: dict = {}


class World:
    """Pre-drawn randomness for one seed (common random numbers across policies)."""

    def __init__(self, seed: int, K: int, T: int):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.fam = rng.integers(0, K, size=T)
        self.u_task = rng.random(T)
        self.u_route = rng.random(T)
        self.v0_z = rng.standard_normal(K)
        self.c_type = rng.random((K, NMAX))
        self.c_mag = rng.random((K, NMAX))
        self.c_shift = rng.standard_normal((K, NMAX))
        self.c_judge = rng.random((K, NMAX))
        self.c_tie = rng.random((K, NMAX))
        self.c_verify = rng.random((K, NMAX, VMAX)).astype(np.float32)


def get_world(seed: int, K: int, T: int) -> World:
    key = (seed, K, T)
    if key not in _WORLD_CACHE:
        if len(_WORLD_CACHE) > 24:
            _WORLD_CACHE.clear()
        _WORLD_CACHE[key] = World(seed, K, T)
    return _WORLD_CACHE[key]


def headroom(v: float, cfg: Config) -> float:
    if not cfg.diminishing:
        return 1.0
    return min(1.0, max(0.05, (cfg.v_max - v) / (cfg.v_max - cfg.v0_mean)))


def mixture_probs(cfg: Config):
    pw, pt, pb = cfg.p_worse, cfg.p_tie, cfg.p_better * cfg.q
    s = pw + pt + pb
    return pw / s, pt / s, pb / s


def draw_delta(cfg: Config, v_parent: float, u_type: float, u_mag: float) -> float:
    pw, pt, _ = mixture_probs(cfg)
    e = -math.log(max(u_mag, 1e-12))
    if u_type < pw:
        return -cfg.worse_scale * e
    if u_type < pw + pt:
        return 0.0
    return cfg.better_scale * headroom(v_parent, cfg) * e


# ---------------------------------------------------------------- data classes
@dataclass(frozen=True)
class ObsView:
    id: str
    n: int            # live outcomes in window
    w: int            # live successes in window
    vk: int           # verification rollouts so far
    vs: int           # verification successes so far
    judge: tuple      # judge verdicts so far (True = judged better than incumbent)
    age: int          # opportunities this candidate has been in flight
    last_fail: bool


@dataclass(frozen=True)
class View:
    t: int
    family: int
    inc: ObsView
    cand: "ObsView | None"
    horizon: float    # estimated remaining tasks of this family
    pub: dict         # public costs / settings


@dataclass
class Decision:
    action: str                    # PROPOSE|HOLD or SKIP|ACCEPT|PROBATE|VERIFY|JUDGE
    k: int = 0
    rule: str = ""
    inputs: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    greedy: str = ""
    reason: str = ""


@dataclass
class Skill:
    id: str
    family: int
    parent: "str | None"
    v: float
    created_t: int
    k: int
    window: int = 60
    status: str = "candidate"
    n: int = 0
    w: int = 0
    vk: int = 0
    vs: int = 0
    vidx: int = 0
    judge: list = field(default_factory=list)
    served: int = 0
    age: int = 0
    tokens_spent: float = 0.0
    delta: float = 0.0
    shift: float = 0.0
    noise: "np.ndarray | None" = None
    judge_n: int = 0               # judge queries already asked of this candidate
    on_probation: bool = False      # policy put this candidate on live trial
    probation_target: "int | None" = None   # observations to collect; None = until policy says otherwise

    def __post_init__(self):
        self.obs = deque(maxlen=self.window)


# ---------------------------------------------------------------- simulator
class Simulator:
    def __init__(self, cfg: Config, policy, seed: int, trace: bool = False):
        self.cfg, self.policy, self.seed = cfg, policy, seed
        self.W = get_world(seed, cfg.K, cfg.T)
        self.trace = Trace(enabled=trace, meta={
            "policy": policy.name, "seed": seed, "config": cfg.to_dict()})
        self.pub = {"c_fail": cfg.c_fail, "tau_prop": cfg.tau_prop,
                    "verify_price": cfg.verify_price, "judge_cost": cfg.judge_cost,
                    "judge_acc": cfg.judge_acc, "q_p": cfg.q_p, "K": cfg.K, "T": cfg.T,
                    "opp_every": cfg.opp_every, "window": cfg.window}

    # hidden value of a skill at time t (never exposed to non-oracle policies)
    def value(self, s: Skill, t: int) -> float:
        v = s.v
        if self.cfg.drift_sd > 0:
            if s.noise is None:
                rng = np.random.default_rng([self.seed, s.family, s.k + 1, 12345])
                s.noise = np.cumsum(rng.standard_normal(self.cfg.T)) * self.cfg.drift_sd
            v = v + s.noise[min(max(t, 0), self.cfg.T - 1)] - s.noise[max(s.created_t, 0)]
        return min(max(v, 0.0), 1.0)

    def true_value(self, skill_id: str) -> float:  # oracle only
        return self.value(self.skills[skill_id], self._t)

    def _obsview(self, s: Skill) -> ObsView:
        return ObsView(s.id, len(s.obs), int(sum(s.obs)), s.vk, s.vs, tuple(s.judge), s.age,
                       bool(s.obs) and s.obs[-1] == 0)

    def _view(self, t: int, f: int) -> View:
        share = max(self.fam_count[f], 1) / (t + 1)
        H = share * (self.cfg.T - t - 1)
        c = self.cand[f]
        return View(t, f, self._obsview(self.inc[f]), self._obsview(c) if c else None, H, self.pub)

    # -------------------------------------------------------------- run
    def run(self) -> dict:
        cfg, W, tr, pol = self.cfg, self.W, self.trace, self.policy
        K, T = cfg.K, cfg.T
        self.skills: dict = {}
        self.inc, self.cand = [], [None] * K
        self.prop_count, self.fam_count = [0] * K, [0] * K
        self.led = dict(fails=0, task_tokens=0.0, proposal=0.0, verify=0.0, judge=0.0)
        self.stats = dict(accepts=0, harmful_accepts=0, good_accepts=0, rejects=0,
                          rejected_good=0, explore_actions=0, proposals_yes=0, proposals_no=0)
        self.rejected_tokens = 0.0
        self._t = 0
        if getattr(pol, "oracle", False):
            pol.hidden = self
        pol.reset()
        for f in range(K):
            s = self._make_root(f)
            self.skills[s.id] = s
            self.inc.append(s)
            tr.log_truth(skill_id=s.id, parent_id=None, true_value=s.v, parent_true_value=None, delta=None)
        cum = np.zeros(T)
        total = 0.0
        for t in range(T):
            self._t = t
            f = int(W.fam[t])
            I, C = self.inc[f], self.cand[f]
            if (C is not None and C.on_probation
                    and (C.probation_target is None or C.n < C.probation_target)
                    and W.u_route[t] < cfg.q_p):
                S, arm, rule = C, "candidate", f"probation_share(q_p={cfg.q_p})"
            else:
                S, arm, rule = I, "incumbent", "family_incumbent"
            succ = self._task_success(S, t, f)
            S.obs.append(1 if succ else 0)
            S.n += 1
            S.w += int(succ)
            S.served += 1
            cost = 1.0 + (0.0 if succ else cfg.c_fail)
            self.led["task_tokens"] += 1.0
            self.led["fails"] += 0 if succ else 1
            total += cost
            cum[t] = total
            tr.log("TASK_SERVED", t, f, skill_id=S.id, routing_rule=rule, arm=arm,
                   outcome=int(succ), tokens=1.0, cost=cost)
            self.fam_count[f] += 1
            if self.fam_count[f] % cfg.opp_every == 0:
                total += self._opportunity(t, f)
                cum[t] = total
        # truth finalisation
        total_check = (cfg.c_fail * self.led["fails"] + self.led["task_tokens"] + self.led["proposal"]
                       + self.led["verify"] + self.led["judge"])
        for f in range(K):
            c = self.cand[f]
            if c is not None:
                self.rejected_tokens += c.tokens_spent
        final_v = float(np.mean([self.value(self.inc[f], T - 1) for f in range(K)]))
        res = dict(total_cost=total_check, success_rate=1 - self.led["fails"] / T,
                   final_value=final_v, proposals=sum(self.prop_count),
                   rejected_tokens=self.rejected_tokens, cum_cost=cum, **self.led, **self.stats)
        res["trace"] = tr
        return res

    # -------------------------------------------------------------- outcome hooks
    # The simulated world draws every outcome from the pre-drawn tables in World.  The
    # real-LLM replay (replay.py) overrides these five methods to read recorded outcomes.
    def _make_root(self, f: int) -> Skill:
        cfg = self.cfg
        v0 = min(max(cfg.v0_mean + cfg.v0_sd * self.W.v0_z[f], 0.05), 0.9)
        return Skill(f"f{f}-s0", f, None, v0, -1, -1, cfg.window, "incumbent")

    def _max_proposals(self, f: int) -> int:
        return NMAX

    def _task_success(self, S: Skill, t: int, f: int) -> bool:
        return bool(self.W.u_task[t] < self.value(S, t))

    def _make_candidate(self, f: int, kidx: int, I: Skill, t: int) -> Skill:
        cfg = self.cfg
        vp = self.value(I, t)
        dl = draw_delta(cfg, vp, self.W.c_type[f, kidx], self.W.c_mag[f, kidx])
        vc = min(vp + dl, cfg.v_max) if dl > 0 else max(vp + dl, cfg.v_floor)
        C = Skill(f"f{f}-c{kidx}", f, I.id, vc, t, kidx, cfg.window, "candidate")
        C.delta = vc - vp
        C.shift = float(self.W.c_shift[f, kidx]) * cfg.shift_sd
        return C

    def _verify(self, C: Skill, f: int, t: int, k: int) -> int:
        """Run k held-out verification rollouts of candidate C; return the successes."""
        vc_now = min(max(self.value(C, t) + C.shift, 0.0), 1.0)
        s_ok = 0
        for _ in range(k):
            idx = C.vidx % VMAX
            s_ok += int(self.W.c_verify[f, C.k, idx] < vc_now)
            C.vidx += 1
        return s_ok

    def _judge(self, C: Skill, I: Skill, f: int, t: int) -> bool:
        """One pairwise judgement: True if the judge says C is better than I."""
        vi, vcn = self.value(I, t), self.value(C, t)
        truly_better = vcn > vi if vcn != vi else bool(self.W.c_tie[f, C.k] < 0.5)
        jidx = (C.k + C.judge_n) % NMAX
        return truly_better if self.W.c_judge[f, jidx] < self.cfg.judge_acc else (not truly_better)

    # -------------------------------------------------------------- opportunity
    def _charge(self, t, f, comp, amount, cause, skill):
        self.led[comp] += amount
        skill.tokens_spent += amount
        self.trace.log("COST_CHARGED", t, f, component=comp, amount=amount, caused_by=cause, skill_id=skill.id)

    def _opportunity(self, t: int, f: int) -> float:
        """Returns extra cost (proposal/verify/judge) incurred at this opportunity."""
        cfg, pol, tr = self.cfg, self.policy, self.trace
        extra = 0.0
        C = self.cand[f]
        if C is not None:
            C.age += 1
        proposed, decisions = False, 0
        while True:
            C = self.cand[f]
            if C is None:
                if proposed or self.prop_count[f] >= self._max_proposals(f):
                    break
                d = pol.propose(self._view(t, f))
                chosen = d.action == "PROPOSE"
                self.stats["proposals_yes" if chosen else "proposals_no"] += 1
                self.stats["explore_actions"] += int(chosen)
                seq = tr.log("PROPOSE_DECISION", t, f, chosen=chosen, mode="EXPLORE" if chosen else "EXPLOIT",
                             rule_id=d.rule, rule_inputs=d.inputs, scores=d.scores, reason=d.reason)
                if not chosen:
                    break
                kidx = self.prop_count[f]
                self.prop_count[f] += 1
                I = self.inc[f]
                vp = self.value(I, t)
                C = self._make_candidate(f, kidx, I, t)
                vc = C.v
                self.skills[C.id] = C
                self.cand[f] = C
                proposed = True
                self._charge(t, f, "proposal", cfg.tau_prop, seq, C)
                extra += cfg.tau_prop
                tr.log("SKILL_CREATED", t, f, skill_id=C.id, parent_id=I.id, proposer_call_id=f"prop-{f}-{kidx}",
                       source="simulated edit (no text in simulator)", tokens=cfg.tau_prop)
                tr.log_truth(skill_id=C.id, parent_id=I.id, true_value=vc, parent_true_value=vp, delta=C.delta)
            view = self._view(t, f)
            if C.age >= cfg.max_age:
                d = Decision("SKIP", rule="sim.timeout", reason=f"in flight for {C.age} opportunities; force-rejected")
            else:
                d = pol.decide(view)
            mode = "EXPLORE" if d.action in ("PROBATE", "VERIFY", "JUDGE") else "EXPLOIT"
            self.stats["explore_actions"] += int(mode == "EXPLORE")
            seq = tr.log("ACTION_CHOSEN", t, f, candidate_id=C.id, action=d.action, k=d.k, mode=mode,
                         greedy_action=d.greedy, rule_id=d.rule, rule_inputs=d.inputs, scores=d.scores,
                         reason=d.reason)
            if d.action == "ACCEPT":
                I = self.inc[f]
                good = C.delta > 0
                self.stats["accepts"] += 1
                self.stats["good_accepts" if good else "harmful_accepts"] += 1
                C.status, I.status = "incumbent", "replaced"
                C.on_probation, C.probation_target = False, None
                self.inc[f], self.cand[f] = C, None
                tr.log("SKILL_ACCEPTED", t, f, skill_id=C.id, replaced_id=I.id, rule_id=d.rule,
                       rule_inputs=d.inputs, n_live=C.n, wins_live=C.w, verify_k=C.vk, verify_s=C.vs)
                tr.log_truth(event="accept", skill_id=C.id, delta=C.delta, was_good_decision=good)
                if proposed:
                    break
            elif d.action == "SKIP":
                self.stats["rejects"] += 1
                self.stats["rejected_good"] += int(C.delta > 0.0)
                self.rejected_tokens += C.tokens_spent
                C.status = "rejected"
                C.on_probation, C.probation_target = False, None
                self.cand[f] = None
                tr.log("SKILL_REJECTED", t, f, skill_id=C.id, rule_id=d.rule, rule_inputs=d.inputs,
                       n_live=C.n, wins_live=C.w, verify_k=C.vk, verify_s=C.vs)
                tr.log_truth(event="reject", skill_id=C.id, delta=C.delta, was_good_decision=C.delta <= 0)
                if proposed:
                    break
            elif d.action == "PROBATE":
                C.on_probation = True
                C.probation_target = None if d.k <= 0 else C.n + d.k
                tr.log("PROBATION_UPDATE", t, f, candidate_id=C.id, n_cand=C.n, wins_cand=C.w,
                       n_inc=len(self.inc[f].obs), wins_inc=int(sum(self.inc[f].obs)),
                       target_obs=C.probation_target, posterior=d.inputs)
                break
            elif d.action == "VERIFY":
                decisions += 1
                if decisions > 3:
                    break
                C.on_probation, C.probation_target = False, None
                s_ok = self._verify(C, f, t, d.k)
                C.vk += d.k
                C.vs += s_ok
                tokens = d.k * cfg.verify_price
                self._charge(t, f, "verify", tokens, seq, C)
                extra += tokens
                tr.log("VERIFY_RESULT", t, f, candidate_id=C.id, k=d.k, successes=s_ok, tokens=tokens,
                       total_k=C.vk, total_s=C.vs)
            elif d.action == "JUDGE":
                decisions += 1
                if decisions > 3:
                    break
                C.on_probation, C.probation_target = False, None
                says_better = self._judge(C, self.inc[f], f, t)
                C.judge_n += 1
                C.judge.append(bool(says_better))
                self._charge(t, f, "judge", cfg.judge_cost, seq, C)
                extra += cfg.judge_cost
                tr.log("JUDGE_RESULT", t, f, candidate_id=C.id, verdict_better=bool(says_better), tokens=cfg.judge_cost)
            else:
                raise ValueError(f"unknown action {d.action}")
        return extra
