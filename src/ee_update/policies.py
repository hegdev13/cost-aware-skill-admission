"""Update policies: baselines (fixed rules) and the cost-aware belief controllers.

Controllers assume a population prior over candidate improvements taken from the
*default* Config, not from the run's true config, so regimes that change the proposer
(D) or the verification gap (E) test misspecification.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config
from .sim import Decision, View, ObsView, headroom, mixture_probs

SQ2 = math.sqrt(2.0)
D_GRID = np.linspace(-0.8, 1.0, 181)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / SQ2))


def e_pos(mu, s):
    """E[max(X,0)] for X ~ N(mu, s^2)."""
    if s <= 1e-12:
        return max(mu, 0.0)
    return mu * Phi(mu / s) + s * phi(mu / s)


def e_pos_shift(mu, s, c):
    """E[max(X - c, 0)] for X ~ N(mu, s^2): the expected gain that survives a margin c."""
    if s <= 1e-12:
        return max(mu - c, 0.0)
    u = (mu - c) / s
    return (mu - c) * Phi(u) + s * phi(u)


class Policy:
    name = "base"
    oracle = False

    def reset(self):
        pass

    def propose(self, v: View) -> Decision:
        return Decision("HOLD", rule="none")

    def decide(self, v: View) -> Decision:
        return Decision("SKIP", rule="none")


# ------------------------------------------------------------------ baselines
class Static(Policy):
    name = "static"

    def propose(self, v):
        return Decision("HOLD", rule="static.never_update", reason="never updates")


class AcceptAll(Policy):
    name = "accept_all"

    def propose(self, v):
        return Decision("PROPOSE", rule="accept_all.always_propose", reason="propose at every opportunity")

    def decide(self, v):
        return Decision("ACCEPT", rule="accept_all.accept", reason=f"{v.cand.id}: accept every candidate unchecked")


class SRNLike(Policy):
    """One-slot revision after a failure, no validation (RSI-Tax SRN-style)."""
    name = "srn_like"

    def propose(self, v):
        yes = v.inc.last_fail
        return Decision("PROPOSE" if yes else "HOLD", rule="srn.propose_after_failure",
                        inputs={"last_task_failed": yes},
                        reason="last task of this family failed -> revise" if yes else "last task succeeded -> no revision")

    def decide(self, v):
        return Decision("ACCEPT", rule="srn.overwrite", reason=f"{v.cand.id}: overwrite the note, no validation")


class GreedyJudge(Policy):
    """EvoGuard-style: one cheap pairwise judgement, accept the winner, never revisit."""
    name = "greedy_judge"

    def propose(self, v):
        return Decision("PROPOSE", rule="judge.always_propose", reason="propose at every opportunity")

    def decide(self, v):
        c = v.cand
        if not c.judge:
            return Decision("JUDGE", rule="judge.ask", reason=f"{c.id}: ask the pairwise judge")
        better = c.judge[-1]
        return Decision("ACCEPT" if better else "SKIP", rule="judge.follow_verdict",
                        inputs={"verdict_better": better}, reason=f"{c.id}: judge says {'better' if better else 'not better'}")


class ThresholdPromote(Policy):
    """StepLearn-style: fixed-length trial, promote if the candidate looks supported."""
    name = "threshold_promote"

    def __init__(self, n_min=10):
        self.n_min = n_min

    def propose(self, v):
        return Decision("PROPOSE", rule="thresh.always_propose", reason="propose at every opportunity")

    def decide(self, v):
        c, i = v.cand, v.inc
        if c.n < self.n_min:
            return Decision("PROBATE", rule="thresh.trial", inputs={"n_cand": c.n, "n_min": self.n_min},
                            reason=f"{c.id}: {c.n}/{self.n_min} trial tasks so far")
        cr, ir = c.w / c.n, (i.w / i.n if i.n else 0.0)
        ok = cr > ir
        return Decision("ACCEPT" if ok else "SKIP", rule="thresh.compare_rates",
                        inputs={"cand_rate": cr, "inc_rate": ir, "n_cand": c.n},
                        reason=f"{c.id}: trial rate {cr:.2f} vs incumbent {ir:.2f} -> {'promote' if ok else 'reject'}")


class ProbeGate(Policy):
    """FGSE-style: fixed probe budget, accept only if the lower confidence bound on gain clears eta."""
    name = "probe_gate"

    def __init__(self, k=20, eta=0.03, z=1.645):
        self.k, self.eta, self.z = k, eta, z

    def propose(self, v):
        return Decision("PROPOSE", rule="probe.always_propose", reason="propose at every opportunity")

    def decide(self, v):
        c, i = v.cand, v.inc
        if c.vk == 0:
            return Decision("VERIFY", k=self.k, rule="probe.run_probes", reason=f"{c.id}: run {self.k} held-out probes")
        pc = c.vs / c.vk
        pi = i.w / i.n if i.n else pc
        var = pc * (1 - pc) / c.vk + (pi * (1 - pi) / i.n if i.n else 0.0)
        lcb = pc - pi - self.z * math.sqrt(var)
        ok = lcb >= self.eta
        return Decision("ACCEPT" if ok else "SKIP", rule="probe.lcb_gate",
                        inputs={"cand": pc, "inc": pi, "lcb_gain": lcb, "eta": self.eta},
                        reason=f"{c.id}: gain LCB {lcb:+.3f} vs bar {self.eta} -> {'commit' if ok else 'reject'}")


class AlwaysVerify(Policy):
    name = "always_verify"

    def __init__(self, k=100):
        self.k = k

    def propose(self, v):
        return Decision("PROPOSE", rule="verify.always_propose", reason="propose at every opportunity")

    def decide(self, v):
        c, i = v.cand, v.inc
        if c.vk == 0:
            return Decision("VERIFY", k=self.k, rule="verify.validate", reason=f"{c.id}: validate with {self.k} rollouts")
        pc = c.vs / c.vk
        pi = i.w / i.n if i.n else pc
        ok = pc > pi
        return Decision("ACCEPT" if ok else "SKIP", rule="verify.point_estimate",
                        inputs={"cand": pc, "inc": pi}, reason=f"{c.id}: validated {pc:.2f} vs incumbent {pi:.2f}")


# ------------------------------------------------------------------ oracle
class Oracle(Policy):
    """Free-information oracle: knows true values, tests for free, proposes only while worth it."""
    name = "oracle"
    oracle = True
    hidden = None

    def _cfg(self):
        return self.hidden.cfg

    def propose(self, v):
        cfg = self._cfg()
        I = self.hidden.skills[v.inc.id]
        vi = self.hidden.value(I, v.t)
        _, _, pb = mixture_probs(cfg)
        gain = pb * cfg.better_scale * headroom(vi, cfg)
        worth = cfg.c_fail * v.horizon * gain
        yes = worth > cfg.tau_prop
        return Decision("PROPOSE" if yes else "HOLD", rule="oracle.expected_gain",
                        inputs={"expected_gain_cost": worth, "tau_prop": cfg.tau_prop},
                        reason="oracle: expected gain of a proposal %s its cost" % (">" if yes else "<="))

    def decide(self, v):
        vc = self.hidden.true_value(v.cand.id)
        vi = self.hidden.true_value(v.inc.id)
        ok = vc > vi
        return Decision("ACCEPT" if ok else "SKIP", rule="oracle.true_value", inputs={"v_cand": vc, "v_inc": vi},
                        reason="oracle sees true values")


# ------------------------------------------------------------------ belief controllers
class BeliefController(Policy):
    """Cost-aware belief controller over D = v_cand - v_inc.

    Belief: a grid posterior over D = v_cand - v_inc.
      prior    the population mixture over rewrite deltas (EvoGuard-calibrated): worse /
               tie / better with exponential magnitudes, scaled by the headroom left above
               the incumbent.  A power prior with weight `prior_weight` flattens it.
      evidence the candidate's own live outcomes (Bernoulli on v_c), its paid verification
               rollouts (Bernoulli on v_c + held-out shift), and pairwise judge verdicts
               (correct with probability `judge_acc`, a step likelihood in D).

    Action values are expected cost savings against "keep the incumbent", in cost units
    (one task rollout = 1, a failure costs c_fail):
      SKIP    0
      ACCEPT  c_fail * H * mu                       (expected gain of swapping now)
      PROBATE c_fail * Hn * e_pos(mu, pre) + n * c_fail * mu
      VERIFY  c_fail * H  * e_pos(mu, pre) - k * verify_price
      JUDGE   c_fail * H  * E[gain | verdict] - judge_cost
    `pre` is how far the posterior mean can still move once the evidence is in
    (pre^2 = var(D) - var(D | evidence)), and e_pos is the expected gain that survives
    the decision to act on the new belief.  PROBATE spends n of the family's tasks on the
    candidate, so the incumbent's horizon drops by n/q_p; VERIFY buys off-task rollouts and
    leaves the horizon intact.
    """

    PROBE_N = (3, 8, 16, 32)
    VERIFY_K = (10, 20, 40, 80)

    def __init__(self, name="c1_voi", z=0.0, gate_propose=False, allow_probate=True,
                 allow_verify=True, allow_judge=True, use_info=True, assumed_shift=0.03,
                 prior_weight=1.0):
        self.name = name
        self.z = z
        self.gate_propose = gate_propose
        self.allow_probate, self.allow_verify, self.allow_judge = allow_probate, allow_verify, allow_judge
        self.use_info = use_info
        self.shift = assumed_shift
        self.prior_weight = prior_weight
        self.A = Config()  # assumed population (not the run's true config)

    # --- prior -------------------------------------------------------------
    def _prior_log(self, v_hat: float) -> np.ndarray:
        A = self.A
        pw, pt, pb = mixture_probs(A)
        hf = headroom(v_hat, A)
        d = D_GRID
        step = float(D_GRID[1] - D_GRID[0])
        dens = np.zeros_like(d)
        neg = d < 0
        dens[neg] = pw * np.exp(d[neg] / A.worse_scale) / A.worse_scale
        pos = d > 0
        scale = max(A.better_scale * hf, 1e-4)
        dens[pos] = pb * np.exp(-d[pos] / scale) / scale
        # the tie component is a point mass at D = 0; on a density grid it is the
        # mass divided by the cell width
        dens[d == 0] = pt / step
        return np.log(dens + 1e-300)

    def _inc_belief(self, o: ObsView):
        m0, sd0 = 0.5, 0.25
        p = min(max(m0, 0.05), 0.95)
        r = p * (1 - p)
        prec, num = 1 / sd0 ** 2, m0 / sd0 ** 2
        if o.n:
            pe = min(max(o.w / o.n, 0.05), 0.95)
            re = pe * (1 - pe)
            prec += o.n / re
            num += o.w / re
        if o.vk:
            pe = min(max(o.vs / o.vk, 0.05), 0.95)
            rv = pe * (1 - pe) / o.vk + self.shift ** 2
            prec += 1 / rv
            num += (o.vs / o.vk) / rv
        return num / prec, 1 / prec

    # --- posterior ---------------------------------------------------------
    def _log_weights(self, o: ObsView, im: float) -> np.ndarray:
        lw = self.prior_weight * self._prior_log(im)
        v = np.clip(im + D_GRID, 1e-6, 1 - 1e-6)
        if o.n:
            lw = lw + o.w * np.log(v) + (o.n - o.w) * np.log(1 - v)
        if o.vk:
            ve = np.clip(v + self.shift, 1e-6, 1 - 1e-6)
            lw = lw + o.vs * np.log(ve) + (o.vk - o.vs) * np.log(1 - ve)
        if o.judge:
            a = self.A.judge_acc
            like = np.where(D_GRID > 0, a, np.where(D_GRID < 0, 1 - a, 0.5))
            for verdict in o.judge:
                lw = lw + np.log(like if verdict else (1.0 - like) + 1e-12)
        lw = lw - lw.max()
        w = np.exp(lw)
        return w / w.sum()

    def _belief(self, o: ObsView, im: float):
        w = self._log_weights(o, im)
        mu = float((w * D_GRID).sum())
        var = float((w * (D_GRID - mu) ** 2).sum())
        return w, mu, var

    def _accept_gain(self, w: np.ndarray, mu: float, var_c: float, iv: float) -> float:
        """Expected cost saving from accepting, in cost units (zero if not worth it)."""
        s = math.sqrt(var_c + iv)
        return self.z * 0.0 + (mu - self.z * s)

    def _trial_expected_gain(self, w_prior, im, n, shift=0.0):
        """Exact expected gain from buying n Bernoulli observations of the candidate.

        The future data are a count w ~ Binomial(n, v_cand).  For every possible count
        the posterior over D is recomputed on the grid, the controller's acceptance
        rule is applied to it, and the realised gain D is counted only when the
        candidate would be accepted.  Nothing is approximated by a Gaussian here.

        Returns E[ D * 1{accept after the evidence} ].
        """
        v = np.clip(im + D_GRID + shift, 1e-6, 1 - 1e-6)
        counts = np.arange(n + 1)
        logp = np.outer(np.log(v), counts) + np.outer(np.log(1 - v), n - counts)
        LL = logp - logp.max(axis=1, keepdims=True)
        M = w_prior[:, None] * np.exp(LL)
        post = M / M.sum(axis=0, keepdims=True)          # posterior over D for each count
        mu_post = post.T @ D_GRID
        var_post = post.T @ (D_GRID ** 2) - mu_post ** 2
        accept = (mu_post - self.z * np.sqrt(np.maximum(var_post, 0.0))) > 0
        P = np.exp(LL)
        P /= P.sum(axis=1, keepdims=True)                 # P(count | truth), per truth
        p_accept = P @ accept.astype(float)
        return float((w_prior * D_GRID * p_accept).sum())

    def _judge_expected_gain(self, w_prior):
        """Exact expected gain from one pairwise judgement."""
        a = self.A.judge_acc
        like = np.where(D_GRID > 0, a, np.where(D_GRID < 0, 1 - a, 0.5))
        total = np.zeros_like(D_GRID)
        for verdict in (True, False):
            lw = np.log(w_prior + 1e-300) + np.log(like if verdict else (1 - like) + 1e-12)
            lw = lw - lw.max()
            wb = np.exp(lw)
            wb /= wb.sum()
            mu_b = float(wb @ D_GRID)
            var_b = float(wb @ (D_GRID ** 2)) - mu_b ** 2
            take = (mu_b - self.z * math.sqrt(max(var_b, 0.0))) > 0
            total += (like if verdict else (1 - like)) * D_GRID * take
        return float((w_prior * total).sum())

    # --- action values -----------------------------------------------------
    def _scores(self, v: View, c: ObsView | None):
        pub, H = v.pub, v.horizon
        cf, q_p = pub["c_fail"], pub["q_p"]
        im, iv = self._inc_belief(v.inc)
        o = c if c is not None else ObsView("none", 0, 0, 0, 0, (), 0, False)
        w, mu, var_c = self._belief(o, im)
        s = math.sqrt(var_c + iv)
        sc = {"SKIP": 0.0, "ACCEPT": cf * H * (mu - self.z * s)}
        info = dict(mu_D=mu, s_D=s, p_better=float(w[D_GRID > 0].sum()), inc_mean=im,
                    inc_sd=math.sqrt(iv), horizon=H, best_probe_n=None, best_verify_k=None)
        if not self.use_info:
            return sc, info

        if self.allow_judge:
            gain = self._judge_expected_gain(w)
            sc["JUDGE"] = cf * H * gain - pub["judge_cost"]
            info["judge_info"] = dict(expect_gain=gain)
        if self.allow_probate:
            best = (-1e18, None, None)
            for n in self.PROBE_N:
                Hn = max(H - n / q_p, 0.0)
                gain = self._trial_expected_gain(w, im, n)
                val = cf * Hn * gain + n * cf * mu
                if val > best[0]:
                    best = (val, n, dict(expect_gain=gain))
            sc["PROBATE"] = best[0]
            info["best_probe_n"] = best[1]
            info["probe_info"] = best[2]
        if self.allow_verify:
            best = (-1e18, None, None)
            for k in self.VERIFY_K:
                gain = self._trial_expected_gain(w, im, k, shift=self.shift)
                val = cf * H * gain - k * pub["verify_price"]
                if val > best[0]:
                    best = (val, k, dict(expect_gain=gain))
            sc["VERIFY"] = best[0]
            info["best_verify_k"] = best[1]
            info["verify_info"] = best[2]
        return sc, info

    # --- decisions ---------------------------------------------------------
    def propose(self, v: View) -> Decision:
        if not self.gate_propose:
            return Decision("PROPOSE", rule=f"{self.name}.propose_always", reason="propose at every opportunity")
        sc, info = self._scores(v, None)
        value = max(sc.values())
        yes = value > v.pub["tau_prop"]
        return Decision("PROPOSE" if yes else "HOLD", rule=f"{self.name}.voi_propose", inputs=info, scores=sc,
                        reason=("best net value of testing a fresh candidate %.1f %s proposal cost %.1f (horizon %.0f tasks, "
                                "incumbent %.2f) -> %s") % (value, ">" if yes else "<=", v.pub["tau_prop"], v.horizon,
                                                            info["inc_mean"], "PROPOSE" if yes else "HOLD"))

    def decide(self, v: View) -> Decision:
        c = v.cand
        sc, info = self._scores(v, c)
        best = max(sc, key=lambda a: (sc[a], a == "SKIP"))
        greedy = "ACCEPT" if sc["ACCEPT"] > 0 else "SKIP"
        k = 0
        if best == "PROBATE":
            k = info["best_probe_n"] or 0
        elif best == "VERIFY":
            k = info["best_verify_k"] or 0
        txt = ", ".join(f"{a} {x:+.1f}" for a, x in sc.items())
        reason = (f"{c.id}: belief D={info['mu_D']:+.3f}±{info['s_D']:.3f} (P(better)={info['p_better']:.2f}), "
                  f"{info['horizon']:.0f} tasks left. Net value: {txt} -> {best}"
                  + (" (explore: buys information)" if best in ("PROBATE", "VERIFY", "JUDGE") else ""))
        return Decision(best, k=k, rule=f"{self.name}.expected_net_value", inputs=info, scores=sc,
                        greedy=greedy, reason=reason)


def make_policy(name: str) -> Policy:
    table = {
        "static": Static, "accept_all": AcceptAll, "srn_like": SRNLike, "greedy_judge": GreedyJudge,
        "threshold_promote": ThresholdPromote, "probe_gate": ProbeGate, "always_verify": AlwaysVerify,
        "oracle": Oracle,
        "c1_voi": lambda: BeliefController("c1_voi"),
        "c3_voi_propose": lambda: BeliefController("c3_voi_propose", gate_propose=True),
        "c3_greedy_mean": lambda: BeliefController("c3_greedy_mean", use_info=False),
        "c3_probe_only": lambda: BeliefController("c3_probe_only", allow_verify=False),
        "c3_verify_only": lambda: BeliefController("c3_verify_only", allow_probate=False),
        "c3_no_judge": lambda: BeliefController("c3_no_judge", allow_judge=False),
        "c1_conservative": lambda: BeliefController("c1_conservative", z=1.0),
        "c1_risk_averse": lambda: BeliefController("c1_risk_averse", z=2.0),
        "c1_flat_prior": lambda: BeliefController("c1_flat_prior", prior_weight=0.25),
    }
    return table[name]()


BASELINES = ["static", "accept_all", "srn_like", "greedy_judge", "threshold_promote", "probe_gate", "always_verify"]
CONTROLLERS = ["c1_voi", "c3_voi_propose"]
ABLATIONS = ["c3_greedy_mean", "c3_probe_only", "c3_verify_only", "c3_no_judge",
             "c1_conservative", "c1_risk_averse", "c1_flat_prior"]
