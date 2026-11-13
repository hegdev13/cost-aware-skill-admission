"""Simulator configuration. Defaults follow docs/04_simulator_spec.md (plus amendments listed there)."""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class Config:
    # world
    K: int = 6                  # task families
    T: int = 1000               # real tasks in the stream
    opp_every: int = 5          # an update opportunity every n-th task of a family
    v0_mean: float = 0.5        # empty-skill success prob, mean over families
    v0_sd: float = 0.15
    v_max: float = 0.95         # ceiling for improvements
    v_floor: float = 0.02
    diminishing: bool = True    # improvements shrink near the ceiling
    # candidate value mixture (EvoGuard-calibrated, see spec 2.2)
    p_worse: float = 0.422
    p_tie: float = 0.077
    p_better: float = 0.500
    worse_scale: float = 0.0587
    better_scale: float = 0.0672
    q: float = 1.0              # proposer quality multiplier on p_better
    drift_sd: float = 0.0       # per-task random-walk sd of each skill's value
    # costs (unit: one task rollout = 1)
    c_fail: float = 5.0
    tau_prop: float = 1.0
    verify_price: float = 1.0   # per held-out verification rollout
    judge_cost: float = 0.1     # a pairwise LLM comparison costs ~10% of a task rollout
    judge_acc: float = 0.7
    shift_sd: float = 0.0       # generalisation gap of held-out verification
    # observation / routing
    q_p: float = 0.2            # share of a family's tasks routed to a candidate on probation
    window: int = 60            # outcomes used to estimate an incumbent
    max_age: int = 40           # opportunities before a candidate in flight is force-rejected

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


# Named regimes (spec section 10, amended): each is a set of overrides on Config().
REGIMES = {
    "base":          {},
    "A_cheap_verify": {"verify_price": 0.2},
    "B_costly_verify": {"verify_price": 2.0},
    "C_costly_fail":  {"c_fail": 20.0},
    "D_weak_proposer": {"q": 0.5},
    "E_gen_gap":      {"shift_sd": 0.08},
    "F_drift":        {"drift_sd": 0.01},
    "G_dear_judge":   {"judge_cost": 1.0},
    "H_noisy_judge":  {"judge_acc": 0.55},
    "J_no_judge":     {"judge_cost": 100.0},
}


def regime(name: str, **extra) -> Config:
    return Config().with_(**{**REGIMES[name], **extra})
