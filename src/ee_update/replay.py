"""Replay every policy on recorded real-LLM outcomes (docs/06_replay_table_spec.md, section 9).

The replay table (data/replay/, built by experiments/build_replay_table.py) holds, per task
family, an empty skill and N candidate skills written by an LLM proposer, the outcome and
token count of every skill on every one of the family's 40 tasks, and a pairwise judge
verdict for every ordered pair of skills.  ReplaySimulator runs the ordinary decision loop
(sim.Simulator) but reads every outcome from that table:

  live task      the incumbent's (or trial candidate's) recorded outcome on the next of the
                 family's 30 stream tasks (seed-shuffled order, cycling)
  new candidate  the next unused candidate of the family (seed-shuffled order); candidates
                 are rewrites of the empty skill, so the proposer does not see the incumbent
  verification   recorded outcomes on the family's 10 held-out tasks (sampled with
                 replacement per candidate)
  judge          the recorded verdict for (candidate, incumbent), alternating the two
                 presentation orders on repeated questions
  true value     success rate over all 40 tasks

Costs are charged in the simulator's rollout-equivalent units (identical rules for every
policy) and, in parallel, in the tokens the recorded calls actually used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .policies import make_policy
from .sim import Simulator, Skill

N_STREAM, N_HELD_OUT = 30, 10


@dataclass
class ReplayTable:
    skills: dict                                   # id -> skill record
    cells: dict                                    # (skill id, task index) -> cell record
    judge: dict                                    # (a id, b id) -> "A" | "B" | "?"
    families: int
    judge_tokens: dict = field(default_factory=dict)  # (a id, b id) -> tokens of that call
    roots: dict = field(default_factory=dict)      # family -> root id
    candidates: dict = field(default_factory=dict)  # family -> [candidate ids]

    @classmethod
    def load(cls, path: str | Path) -> "ReplayTable":
        d = Path(path)
        skills = {s["id"]: s for s in map(json.loads, open(d / "skills.jsonl"))}
        cells = {}
        for c in map(json.loads, open(d / "cells.jsonl")):
            cells[(c["skill_id"], int(c["task_id"].split("-t")[1]))] = c
        rows = [json.loads(x) for x in open(d / "judge.jsonl")]
        judge = {(j["a"], j["b"]): j["says"] for j in rows}
        fams = sorted({s["family"] for s in skills.values()})
        t = cls(skills, cells, judge, len(fams))
        t.judge_tokens = {(j["a"], j["b"]): j["prompt_tokens"] + j["output_tokens"] for j in rows}
        for f in fams:
            t.roots[f] = next(s["id"] for s in skills.values() if s["family"] == f and s["parent_id"] is None)
            t.candidates[f] = sorted((s["id"] for s in skills.values() if s["family"] == f and s["parent_id"]),
                                     key=lambda i: int(i.split("-c")[1]))
        return t

    def value(self, sid: str) -> float:
        return self.skills[sid]["true_value"]

    def mean_task_tokens(self) -> float:
        return float(np.mean([c["prompt_tokens"] + c["output_tokens"] for c in self.cells.values()]))

    def measured_prices(self) -> dict:
        """Proposal, verification-probe and judge prices in task-rollout units, from the tokens
        the recorded calls actually used (one task rollout = the mean agent-task tokens)."""
        task = self.mean_task_tokens()
        prop = np.mean([s["proposer_prompt_tokens"] + s["proposer_output_tokens"]
                        for s in self.skills.values() if s["parent_id"]])
        held = np.mean([c["prompt_tokens"] + c["output_tokens"] for (sid, i), c in self.cells.items() if i >= N_STREAM])
        judge = np.mean(list(self.judge_tokens.values()))
        return dict(tau_prop=float(prop / task), verify_price=float(held / task), judge_cost=float(judge / task))


def replay_config(**kw) -> Config:
    """The base-regime prices on the replay's world: 5 families, 120 tasks each.  Pass
    **table.measured_prices() to charge every action what the recorded calls actually cost."""
    base = dict(K=5, T=600, drift_sd=0.0, shift_sd=0.0)
    base.update(kw)
    return Config().with_(**base)


class ReplaySimulator(Simulator):
    def __init__(self, cfg: Config, policy, seed: int, table: ReplayTable, trace: bool = False):
        assert cfg.K == table.families, "config K must equal the number of families in the table"
        super().__init__(cfg, policy, seed, trace=trace)
        self.table = table
        rng = np.random.default_rng([seed, 4242])
        self.stream = {f: rng.permutation(N_STREAM) for f in range(cfg.K)}
        self.cand_order = {f: list(rng.permutation(table.candidates[f])) for f in range(cfg.K)}
        self.tok = dict(task=0.0, proposal=0.0, verify=0.0, judge=0.0)

    # --- hooks -------------------------------------------------------------
    def _make_root(self, f: int) -> Skill:
        tid = self.table.roots[f]
        s = Skill(f"f{f}-s0", f, None, self.table.value(tid), -1, -1, self.cfg.window, "incumbent")
        s.tid = tid
        return s

    def _max_proposals(self, f: int) -> int:
        return len(self.table.candidates[f])

    def _task_success(self, S: Skill, t: int, f: int) -> bool:
        task = int(self.stream[f][self.fam_count[f] % N_STREAM])
        c = self.table.cells[(S.tid, task)]
        self.tok["task"] += c["prompt_tokens"] + c["output_tokens"]
        return bool(c["success"])

    def _make_candidate(self, f: int, kidx: int, I: Skill, t: int) -> Skill:
        tid = self.cand_order[f][kidx]
        rec = self.table.skills[tid]
        C = Skill(f"f{f}-c{kidx}", f, I.id, self.table.value(tid), t, kidx, self.cfg.window, "candidate")
        C.tid = tid
        C.delta = C.v - I.v
        C.ho_rng = np.random.default_rng([self.seed, f, kidx, 99])
        self.tok["proposal"] += rec["proposer_prompt_tokens"] + rec["proposer_output_tokens"]
        return C

    def _verify(self, C: Skill, f: int, t: int, k: int) -> int:
        ok = 0
        for task in N_STREAM + C.ho_rng.integers(0, N_HELD_OUT, size=k):
            c = self.table.cells[(C.tid, int(task))]
            ok += c["success"]
            self.tok["verify"] += c["prompt_tokens"] + c["output_tokens"]
        return int(ok)

    def _judge(self, C: Skill, I: Skill, f: int, t: int) -> bool:
        pair = (C.tid, I.tid) if C.judge_n % 2 == 0 else (I.tid, C.tid)
        self.tok["judge"] += self.table.judge_tokens[pair]
        return self.table.judge[pair] == ("A" if pair[0] == C.tid else "B")

    # --- run ---------------------------------------------------------------
    def run(self) -> dict:
        res = super().run()
        per_task = self.table.mean_task_tokens()
        res["tokens"] = dict(self.tok)
        res["token_cost"] = (self.cfg.c_fail * res["fails"] * per_task + sum(self.tok.values()))
        return res


def run_replay(table: ReplayTable, cfg: Config, policy: str, seed: int, trace: bool = False) -> dict:
    return ReplaySimulator(cfg, make_policy(policy), seed, table, trace=trace).run()
