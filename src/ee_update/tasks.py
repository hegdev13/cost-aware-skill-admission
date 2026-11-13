"""Task suite for the real-LLM stage (docs/06_replay_table_spec.md, section 3).

Five families of single-turn tasks.  Each family hides one deterministic convention
that is never stated in the prompt: a skill that states the convention turns that
family's failures into successes, which is what makes the explore/exploit mechanics
visible in real model behaviour.

Every task has a programmatic verifier - exact string match after normalisation,
never an LLM judge - and a failure diff of the shape "expected: X | got: Y", which
is what a proposer reads to infer the rule.

The suite is deterministic: the same family always yields the same tasks, so the
replay table is reproducible and two runs are comparable.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

N_FAMILIES = 5
N_STREAM = 30          # tasks the policy sees live
N_HELD_OUT = 10        # tasks used only for verification and for the truth
N_TASKS = N_STREAM + N_HELD_OUT    # = 40
N_FREE = 16            # per family: tasks that need no convention (keeps the empty
                       # skill's success rate comfortably in the 25-60% band)

FAMILY_NAMES = ["record dates", "ticket ids", "unit conversion", "json records",
                "field redaction"]
FAMILY_CONVENTIONS = [
    "our records use day-first order, zero-padded day and month, four-digit year: 17/03/2024",
    "ticket ids are PREFIX-SEQUENCE-CHECK, four-digit sequence, check digit = sum of the "
    "four sequence digits modulo 10: AB-0427-3",
    "answers are rounded to one decimal place, half away from zero, and carry no unit: 37.4",
    "records are written as JSON with snake_case keys sorted alphabetically and indented "
    "with two spaces",
    "the values of the fields id, email, dob and phone are replaced by the literal "
    "REDACTED; every other field is kept, in the original order",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


@dataclass(frozen=True)
class Task:
    id: str
    family: int
    held_out: bool
    prompt: str
    expected: str


# ----------------------------------------------------------------- families
def _round_half_away(x: float, nd: int = 1) -> str:
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def _f0_dates(i: int, rnd: random.Random) -> tuple[str, str]:
    """Family 0: dates. Convention: day-first, zero-padded, four-digit year."""
    if i < N_FREE:
        # Free task: pure arithmetic, no convention needed
        a, b = sorted((rnd.randint(1, 20), rnd.randint(21, 60)))
        return (f"How many days are there between {a}/01/2024 and {b}/01/2024? "
                f"Answer with the number only.", str(b - a))
    day, month, year = rnd.randint(1, 28), rnd.randint(1, 12), rnd.randint(2020, 2026)
    # Prompt does NOT state the convention; the model must infer the format from the
    # task description (which is deliberately vague).  The expected answer uses day-first.
    return (f"Record date: {day} {MONTHS[month - 1]} {year}. Format it for the records "
            f"system. Answer with the date only.", f"{day:02d}/{month:02d}/{year}")


def _f1_tickets(i: int, rnd: random.Random) -> tuple[str, str]:
    """Family 1: ticket ids. Convention: PREFIX-SEQUENCE-CHECK, check = sum digits mod 10."""
    if i < N_FREE:
        # Free task: arithmetic, model always gets these right
        a, b = rnd.randint(12, 99), rnd.randint(12, 99)
        return (f"What is {a} times {b}? Answer with the number only.", str(a * b))
    prefix = rnd.choice(["AB", "CD", "EF", "GH", "JK", "LM", "NP", "QR"])
    seq = rnd.randint(1000, 9999)
    check = sum(int(c) for c in str(seq)) % 10
    # Prompt does NOT state the convention.  The model only sees the prompt and must
    # produce a ticket id; without knowing the checksum rule it will usually omit it
    # or guess a wrong digit.
    return (f"Issue the ticket for project {prefix}, sequence {seq}. Answer with the "
            f"ticket id only.", f"{prefix}-{seq}-{check}")


_CONVERSIONS = {
    ("celsius", "fahrenheit"): (lambda c: c * 9 / 5 + 32, lambda f: (f - 32) * 5 / 9, 1.8),
    ("metres", "feet"): (lambda m: m * 3.28084, lambda f: f / 3.28084, 1.0),
    ("kilograms", "pounds"): (lambda k: k * 2.20462, lambda p: p / 2.20462, 2.0),
    ("litres", "gallons"): (lambda l: l * 0.264172, lambda g: g / 0.264172, 3.0),
}


def _f2_units(i: int, rnd: random.Random) -> tuple[str, str]:
    """Family 2: unit conversion. Convention: round to ONE decimal place, half away from zero."""
    if i < N_FREE:
        # Free task: addition, model always gets these right
        a, b = rnd.randint(12, 99), rnd.randint(12, 99)
        return (f"What is {a} plus {b}? Answer with the number only.", str(a + b))
    (u1, u2), (fwd, inv, _prec) = rnd.choice(list(_CONVERSIONS.items()))
    value = round(rnd.uniform(3.0, 90.0), rnd.choice([1, 2, 3]))
    if rnd.random() < 0.5:
        out = fwd(value)
    else:                       # the reverse direction: the prompt must say so too
        out = inv(value)
        u1, u2 = u2, u1
    # Convention: ONE decimal place, half away from zero.  The prompt does NOT state
    # the rounding rule; a model that normally rounds to 2 dp or uses banker's rounding
    # will often produce a different answer.
    return (f"Convert {value} {u1} to {u2}. Answer with the number only.",
            _round_half_away(out, 1))


def _f3_json(i: int, rnd: random.Random) -> tuple[str, str]:
    """Family 3: json records. Convention: snake_case keys, sorted alphabetically, 2-space indent."""
    if i < N_FREE:
        # Free task: count fields, model always gets these right
        rec = {"orderId": rnd.randint(1000, 9999), "totalAmount": rnd.randint(10, 900)}
        return (f"How many fields does this record have? Answer with the number only.\n"
                + json.dumps(rec), str(len(rec)))
    rec = {"orderId": rnd.randint(1000, 9999),
           "customerName": f"customer-{rnd.randint(1, 99)}",
           "totalAmount": rnd.randint(10, 900),
           "isPaid": rnd.choice([True, False])}
    snake = {("order_id" if k == "orderId" else
              "customer_name" if k == "customerName" else
              "total_amount" if k == "totalAmount" else
              "is_paid"): v for k, v in rec.items()}
    expected = json.dumps(snake, indent=2, sort_keys=True)
    # Prompt does NOT state the convention (key naming, sort order, indent width).  The
    # model typically outputs JSON with camelCase or unsorted keys and no indent, so the
    # canonical form almost never matches without the skill.
    return ("Rewrite this record as JSON for the API. Answer with the JSON only.\n"
            + json.dumps(rec), expected)


def _f4_redaction(i: int, rnd: random.Random) -> tuple[str, str]:
    """Family 4: field redaction. Convention: replace id/email/dob/phone with REDACTED."""
    if i < N_FREE:
        # Free task: character count, model always gets these right
        text = f"note-{rnd.randint(100, 999)}"
        return (f"How many characters are in this note? Answer with the number only.\n{text}",
                str(len(text)))
    fields = [("id", str(rnd.randint(1000, 9999))), ("email", f"u{rnd.randint(1, 99)}@example.com"),
              ("dob", f"{rnd.randint(1950, 2005)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"),
              ("phone", f"555-{rnd.randint(1000, 9999)}"),
              ("plan", rnd.choice(["basic", "plus", "pro"])),
              ("region", rnd.choice(["emea", "apac", "amer"]))]
    rnd.shuffle(fields)
    redact = {"id", "email", "dob", "phone"}
    lines_in = "\n".join(f"{k}: {v}" for k, v in fields)
    lines_out = "\n".join(f"{k}: {'REDACTED' if k in redact else v}" for k, v in fields)
    # Prompt does NOT state which fields to redact.  A model that redacts a different
    # set (or spells REDACTED differently) will fail the exact-match verifier.
    return (f"Produce the redacted record. Answer with the record only.\n{lines_in}", lines_out)


_BUILDERS = [_f0_dates, _f1_tickets, _f2_units, _f3_json, _f4_redaction]


def family_tasks(family: int, seed: int = 20261113) -> list[Task]:
    """All 40 tasks of one family: 30 stream tasks then 10 held-out tasks."""
    rnd = random.Random(seed + 1000 * family)
    tasks = []
    for i in range(N_TASKS):
        prompt, expected = _BUILDERS[family](i, rnd)
        tasks.append(Task(id=f"f{family}-t{i:02d}", family=family,
                          held_out=i >= N_STREAM, prompt=prompt, expected=expected))
    return tasks


def all_tasks(seed: int = 20261113) -> list[Task]:
    out = []
    for f in range(N_FAMILIES):
        out.extend(family_tasks(f, seed))
    return out


# ----------------------------------------------------------------- verifier
def _strip_fences(text: str) -> str:
    """Drop a surrounding Markdown code fence (```json ... ```), which chat models add
    even when told to answer with the value only.  Nothing inside the fence is changed."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```") and len(t) >= 6:
        t = t[3:-3]
        first_nl = t.find("\n")
        if first_nl != -1 and t[:first_nl].strip().isalpha():   # language tag
            t = t[first_nl + 1:]
    return t.strip()


def check(family: int, answer: str, expected: str) -> tuple[bool, str]:
    """Programmatic verifier. Returns (ok, diff). No LLM judging anywhere."""
    a = _strip_fences(answer or "")
    e = expected.strip()
    if a == e:
        return True, ""
    if family == 3:   # json: compare canonical forms so trailing whitespace is forgiven
        try:
            pa = json.loads(a)
        except Exception:
            pa = None
        if pa is not None and json.dumps(pa, indent=2, sort_keys=True) == e:
            return True, ""
    return False, f"expected: {e} | got: {a}"


def convention_text(family: int) -> str:
    """The rule a correct skill has to state (kept out of every prompt)."""
    return FAMILY_CONVENTIONS[family]