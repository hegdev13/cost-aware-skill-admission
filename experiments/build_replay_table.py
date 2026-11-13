"""Build the real-LLM replay table (docs/06_replay_table_spec.md, as amended in section 9).

    export GEMINI_API_KEY=...            # never commit it
    uv run python experiments/build_replay_table.py

Per task family (5 families x 40 tasks, src/ee_update/tasks.py):
  1. the empty skill s0 is run on all 40 tasks;
  2. a proposer writes N_CAND candidate skills, each from a different sample of s0's real
     failures on the 30 stream tasks (prompt, expected answer, the model's answer);
  3. every candidate is run on all 40 tasks;
  4. a pairwise judge compares every pair of the family's skills, in both orders.
Agent, proposer and judge are three different Gemini models, all at temperature 0.  To fit
the free tier (15 requests per minute per model) the agent answers BATCH tasks per call as
a JSON list; each task's answer is still checked on its own, and a call's tokens are split
evenly over its tasks.  Responses are cached on disk, so a rerun is free and an interrupted
build resumes.  Writes data/replay/{skills,cells,judge}.jsonl + meta.json.
A skill's true value is its success rate over its family's 40 tasks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ee_update.llm import Gemini  # noqa: E402
from ee_update.tasks import FAMILY_NAMES, N_FAMILIES, N_STREAM, check, family_tasks  # noqa: E402

AGENT_MODEL = "gemini-3.5-flash-lite"
PROPOSER_MODEL = "gemini-3.5-flash-lite"   # the agent rewrites its own notes (free tier: gemini-3.5-flash allows 20 requests/day)
JUDGE_MODEL = "gemini-3.1-flash-lite"
N_CAND = 9
BATCH = 10
JUDGE_EXAMPLES = (0, 16, 17, 18)     # task indices shown to the judge (stream tasks only)


def agent_prompt(skill_text: str, task_prompts: list) -> str:
    body = "\n\n".join(f"Task {i + 1}:\n{p}" for i, p in enumerate(task_prompts))
    return (f"Notes for this kind of task:\n{skill_text.strip() or '(none yet)'}\n\n"
            f"Answer each of the {len(task_prompts)} tasks below on its own. Reply with a JSON list of "
            f"{len(task_prompts)} strings, in task order, each string being exactly what that task asks "
            f"for, and nothing else.\n\n{body}")


def parse_answers(text: str, n: int) -> list:
    """The agent's JSON list -> n answer strings ('' where an answer is missing)."""
    from ee_update.tasks import _strip_fences
    try:
        xs = json.loads(_strip_fences(text))
    except Exception:
        return [""] * n
    if not isinstance(xs, list):
        return [""] * n
    out = [x if isinstance(x, str) else json.dumps(x, indent=2) for x in xs[:n]]
    return out + [""] * (n - len(out))


def proposer_prompt(skill_text: str, failures: list) -> str:
    shown = "\n\n".join(f"Task: {t.prompt}\nExpected answer: {t.expected}\nYour answer: {a}"
                        for t, a in failures)
    return ("You maintain a short note that an assistant reads before solving tasks of one kind. "
            "Here is the current note:\n"
            f"{skill_text.strip() or '(empty)'}\n\n"
            "These tasks were answered wrong (an exact-match checker compares the answer with the "
            f"expected answer):\n\n{shown}\n\n"
            "Rewrite the note so that future tasks of this kind are answered exactly right. "
            "State general rules, not the answers to these particular tasks. "
            "At most 80 words. Output only the new note.")


def judge_prompt(examples: list, note_a: str, note_b: str) -> str:
    ex = "\n".join(f"- {t.prompt}" for t in examples)
    return ("An assistant reads a short note before answering tasks like these; answers are checked "
            f"by exact match against a hidden expected format.\n\nExample tasks:\n{ex}\n\n"
            f"Note A:\n{note_a.strip() or '(empty note)'}\n\nNote B:\n{note_b.strip() or '(empty note)'}\n\n"
            "Which note will make the assistant answer more of these tasks exactly right? "
            "Reply with the single letter A or B.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "replay"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "ee-update" / "gemini"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=14, help="requests per minute, per model")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = Gemini(AGENT_MODEL, args.cache, rpm=args.rpm, max_output_tokens=3000)
    proposer = Gemini(PROPOSER_MODEL, args.cache, rpm=args.rpm, max_output_tokens=2000)
    judge = Gemini(JUDGE_MODEL, args.cache, rpm=args.rpm, max_output_tokens=200)
    pool = ThreadPoolExecutor(args.workers)
    t0 = time.time()

    tasks = {f: family_tasks(f) for f in range(N_FAMILIES)}
    skills: list[dict] = []
    cells: list[dict] = []

    def run_skill(sk: dict) -> list[dict]:
        ts = tasks[sk["family"]]
        batches = [ts[i:i + BATCH] for i in range(0, len(ts), BATCH)]
        replies = list(pool.map(lambda b: agent.ask(agent_prompt(sk["text"], [t.prompt for t in b])), batches))
        rows = []
        for b, r in zip(batches, replies):
            answers = parse_answers(r.text, len(b))
            for t, a in zip(b, answers):
                ok, _ = check(t.family, a, t.expected)
                rows.append(dict(skill_id=sk["id"], task_id=t.id, family=t.family, held_out=t.held_out,
                                 success=int(ok), answer=a, prompt_tokens=r.prompt_tokens / len(b),
                                 output_tokens=r.output_tokens / len(b), response_hash=r.response_hash))
        sk["true_value"] = sum(x["success"] for x in rows) / len(rows)
        sk["stream_value"] = sum(x["success"] for x in rows if not x["held_out"]) / N_STREAM
        sk["held_out_value"] = sum(x["success"] for x in rows if x["held_out"]) / (len(rows) - N_STREAM)
        return rows

    # 1. empty skills
    roots = [dict(id=f"f{f}-s0", family=f, parent_id=None, text="", proposer_prompt_tokens=0,
                  proposer_output_tokens=0, source_task_ids=[]) for f in range(N_FAMILIES)]
    for sk in roots:
        cells += run_skill(sk)
        skills.append(sk)
        print(f"{sk['id']} ({FAMILY_NAMES[sk['family']]}): true value {sk['true_value']:.3f}", flush=True)

    # 2-3. candidates from real failures of the empty skill
    for sk0 in roots:
        f = sk0["family"]
        fails = [(t, c["answer"]) for t, c in zip(tasks[f], [c for c in cells if c["skill_id"] == sk0["id"]])
                 if not t.held_out and not c["success"]]
        rng = random.Random(1000 + f)
        seen = set()
        samples = []
        while len(samples) < N_CAND:
            m = min(len(fails), rng.choice([2, 3, 4]))
            pick = tuple(sorted(rng.sample(range(len(fails)), m)))
            if pick in seen and len(seen) < 200:
                continue
            seen.add(pick)
            samples.append(pick)
        props = list(pool.map(lambda pk: proposer.ask(proposer_prompt("", [fails[i] for i in pk])), samples))
        for k, (pk, r) in enumerate(zip(samples, props)):
            skills.append(dict(id=f"f{f}-c{k}", family=f, parent_id=sk0["id"], text=r.text.strip(),
                               proposer_prompt_tokens=r.prompt_tokens, proposer_output_tokens=r.output_tokens,
                               source_task_ids=[fails[i][0].id for i in pk]))
        print(f"family {f}: {N_CAND} candidates proposed", flush=True)

    # 3 + 4 in parallel (different models, separate quotas): evaluate candidates; judge every
    # ordered pair of the family's skills
    judge_rows = []

    def judge_all():
        for f in range(N_FAMILIES):
            fam = [s for s in skills if s["family"] == f]
            ex = [tasks[f][i] for i in JUDGE_EXAMPLES]
            jobs = list(itertools.permutations(fam, 2))
            replies = list(jpool.map(lambda ab: judge.ask(judge_prompt(ex, ab[0]["text"], ab[1]["text"])), jobs))
            for (a, b), r in zip(jobs, replies):
                letter = r.text.strip().upper()[:1]
                judge_rows.append(dict(family=f, a=a["id"], b=b["id"], says=letter if letter in "AB" else "?",
                                       prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens))
            print(f"family {f}: {len(jobs)} judge calls", flush=True)

    jpool = ThreadPoolExecutor(args.workers)
    jthread = ThreadPoolExecutor(1).submit(judge_all)
    for sk in [s for s in skills if s["parent_id"]]:
        cells += run_skill(sk)
        print(f"{sk['id']}: true value {sk['true_value']:.3f} "
              f"(empty skill {skills[sk['family']]['true_value']:.3f})", flush=True)
    jthread.result()

    def dump(name, rows):
        with open(out / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    dump("skills.jsonl", skills)
    dump("cells.jsonl", cells)
    dump("judge.jsonl", judge_rows)
    meta = dict(agent_model=AGENT_MODEL, proposer_model=PROPOSER_MODEL, judge_model=JUDGE_MODEL,
                agent_tasks_per_call=BATCH, temperature=0.0, families=N_FAMILIES, tasks_per_family=40,
                candidates_per_family=N_CAND, agent_cells=len(cells),
                agent_calls=len(cells) // BATCH,
                proposer_calls=sum(1 for s in skills if s["parent_id"]), judge_calls=len(judge_rows),
                live_calls=agent.live_calls + proposer.live_calls + judge.live_calls,
                cached_calls=agent.cached_calls + proposer.cached_calls + judge.cached_calls,
                seconds=round(time.time() - t0, 1), built=time.strftime("%Y-%m-%d"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
