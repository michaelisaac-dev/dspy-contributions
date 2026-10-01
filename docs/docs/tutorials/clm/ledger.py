"""CLM vs. RLM on a streamed ledger: a small context-management demo.

A ContextBench-style task (Shao et al., 2026): the input arrives as a stream the agent pulls with
`next_batch()`, so whatever the agent reads lands in its context. Each batch mixes ledger events with
distractors written in loose natural language: planned payments, declined requests, net-zero exchanges,
and corrections that cancel a payment from an earlier batch. The last batch asks for some final balances.

Four programs, same model, same tools, same step limit:

- RLM            stock dspy.RLM; its history only grows and has no budget
- RLM @ budget   dspy.RLM held to the CLM's token budget (overflow ends the run, as for CLM)
- CLM            dspy.CLM with no budget: same as RLM except it can rewrite its live context through CONTEXT_FILE
- CLM @ budget   dspy.CLM held to the budget

Run:  python docs/docs/tutorials/clm/ledger.py --model anthropic/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from baselines import run_program

import dspy
from dspy.predict.clm import approx_tokens
from dspy.primitives.prediction import Prediction

NAMES = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Kendall", "Logan", "Morgan", "Quinn"]
WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty", 40: "forty", 50: "fifty"}

# =============================================================================
# Dataset
# =============================================================================


@dataclass
class Episode:
    seed: int
    batches: list[str]
    question_names: list[str]
    answer: dict[str, int]
    cursor: int = field(default=0, repr=False)

    def next_batch(self) -> str:
        """Return the next batch of the ledger stream. The final batch contains the question."""
        if self.cursor >= len(self.batches):
            return "The stream has ended. Answer the question from the final batch."
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch


def make_episode(seed: int, n_people: int = 5, n_batches: int = 12, events_per_batch: int = 5) -> Episode:
    rng = random.Random(seed)
    people = rng.sample(NAMES, n_people)
    balance = {p: rng.randrange(60, 160, 5) for p in people}
    amount = lambda: rng.choice(list(WORDS))  # noqa: E731
    pays: list[tuple[int, str, str, int]] = []  # (batch, payer, payee, amount), cancellable later

    def pair():
        return rng.sample(people, 2)

    def event(batch_no: int) -> str:
        kind = rng.choices(["pay", "cash", "net_zero", "cancel"], weights=[5, 3, 1, 1.5])[0]
        if kind == "cancel":
            earlier = [p for p in pays if p[0] < batch_no]
            if earlier:
                b, a, c, x = rng.choice(earlier)
                pays.remove((b, a, c, x))
                balance[a] += x
                balance[c] -= x
                return f"Correction: the ${x} payment from {a} to {c} in batch {b} bounced. Reverse it."
            kind = "pay"
        if kind == "pay":
            a, c = pair()
            x = amount()
            balance[a] -= x
            balance[c] += x
            pays.append((batch_no, a, c, x))
            return rng.choice([
                f"{a} paid {c} ${x}.",
                f"{c} received ${x} from {a}.",
                f"{a} sent {c} {WORDS[x]} dollars for the concert tickets.",
                f"{a} transferred ${x} to {c}.",
            ])
        if kind == "cash":
            a, x = rng.choice(people), amount()
            if rng.random() < 0.5:
                balance[a] += x
                return rng.choice([f"{a} deposited ${x}.", f"{a} found {WORDS[x]} dollars in an old jacket and deposited it."])
            balance[a] -= x
            return rng.choice([f"{a} withdrew ${x} at the ATM.", f"{a} took out {WORDS[x]} dollars in cash."])
        a, c = pair()
        return f"{a} covered {c}'s ${amount()} share of the groceries, and {c} paid {a} back the same day."

    def chatter() -> str:
        a, c = pair()
        x = amount()
        return rng.choice([
            f"{a} says they will pay {c} ${x} next week (not sent yet).",
            f"{a} asked {c} for ${x}, but {c} said no.",
            f"{a} thinks the dinner cost about ${x}.",
            f"{a} is considering withdrawing ${x} but has not decided.",
            f"{c} reminded {a} about the ${x} they owe from last year (already settled, ignore).",
            f"Weather: {rng.randint(40, 90)} degrees, wind {rng.randint(1, 30)} mph.",
        ])

    def noise() -> str:
        return f"[audit] node={rng.randrange(16**6):06x} seq={rng.randrange(10**8)} checksum={rng.randrange(16**16):016x} ok"

    opening = ", ".join(f"{p} ${balance[p]}" for p in people)
    batches = []
    for b in range(1, n_batches + 1):
        lines = [event(b) for _ in range(events_per_batch)]
        lines += [chatter() for _ in range(6)] + [noise() for _ in range(6)]
        events, rest = lines[:events_per_batch], lines[events_per_batch:]
        rng.shuffle(rest)
        # Events keep their order (a correction must follow the payment it cancels); distractors interleave.
        merged = []
        for line in events:
            k = rng.randint(1, 3)
            merged += [line, *rest[:k]]
            rest = rest[k:]
        merged += rest
        header = f"=== batch {b} ==="
        if b == 1:
            header += f"\nOpening balances: {opening}."
        batches.append(header + "\n" + "\n".join(merged))

    question_names = sorted(rng.sample(people, 3))
    batches.append(
        "=== END OF STREAM ===\n"
        f"Question: what are the final balances of {', '.join(question_names)}? "
        "Return a dict mapping each of these names to an integer number of dollars."
    )
    return Episode(seed, batches, question_names, {p: balance[p] for p in question_names})


TASK = (
    "You are tracking account balances from a stream of ledger updates written in natural language. "
    "Call next_batch() to receive the stream one batch at a time; keep calling it until it reports the end of the stream. "
    "Only things that actually happened change balances: payments, transfers, deposits, withdrawals, and corrections "
    "that reverse an earlier payment. Plans, requests, refusals, estimates, and audit lines change nothing. "
    "The final batch asks a question; answer it exactly."
)


class Ledger(dspy.Signature):
    """Track balances through the ledger stream and answer the final question."""

    task: str = dspy.InputField()
    balances: dict[str, int] = dspy.OutputField(desc="final balance for each name the question asks about")


# =============================================================================
# Evaluation
# =============================================================================


def score(ep: Episode, pred: Prediction | None) -> float:
    got = getattr(pred, "balances", None) or {}
    if not isinstance(got, dict):
        return 0.0
    got = {str(k).strip().lower(): v for k, v in got.items()}
    return sum(got.get(n.lower()) == v for n, v in ep.answer.items()) / len(ep.answer)


def run_one(method: str, seed: int, args) -> dict:
    ep = make_episode(seed)
    t0 = time.time()
    try:
        pred = run_program(method, Ledger, [ep.next_batch], args.max_iters, args.budget, task=TASK)
        error = None
    except Exception as e:  # keep the sweep going; a crash scores zero
        pred, error = None, f"{type(e).__name__}: {e}"
    usage = (pred.get_lm_usage() or {}) if pred is not None else {}
    prompt_tokens = sum(u.get("prompt_tokens", 0) or 0 for u in usage.values())
    stats = getattr(pred, "context_stats", {}) or {}
    return {
        "method": method,
        "seed": seed,
        "score": score(ep, pred),
        "batches_read": min(ep.cursor, len(ep.batches)),
        "n_batches": len(ep.batches),
        "answer": ep.answer,
        "predicted": getattr(pred, "balances", None),
        "stats": stats,
        "prompt_tokens": prompt_tokens,
        "seconds": round(time.time() - t0, 1),
        "error": error,
        "trajectory": getattr(pred, "trajectory", None),
        "final_context": getattr(pred, "final_context", None),
    }


def _mean_stat(rows: list[dict], key: str) -> float:
    return sum(r["stats"].get(key, 0) for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--budget", type=int, default=2000, help="live-context budget in ~tokens (4 chars/token)")
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--methods", default="RLM,CLM,RLM @ budget,CLM @ budget")
    parser.add_argument("--out", default=None, help="write per-run JSON results here")
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM(args.model, max_tokens=8000, temperature=0.0), track_usage=True)
    methods = [m.strip() for m in args.methods.split(",")]

    sample = make_episode(0)
    stream_tokens = sum(approx_tokens(b) for b in sample.batches)
    print(f"Stream: {len(sample.batches)} batches, ~{stream_tokens} tokens of input per episode "
          f"(~{stream_tokens / args.budget:.1f}x the {args.budget}-token budget before any code or reasoning).\n")

    jobs = [(m, s) for m in methods for s in range(args.seeds)]
    with ThreadPoolExecutor(args.threads) as pool:
        rows = list(pool.map(lambda job: run_one(*job, args), jobs))

    header = f"{'method':<14} {'accuracy':>8} {'all-3':>6} {'read all':>8} {'overflow':>8} {'peak ctx':>9} {'prefill (prefix reuse)':>23} {'prompt tok':>11}"
    print(header)
    print("-" * len(header))
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        n = len(rs)
        print(
            f"{m:<14} {sum(r['score'] for r in rs) / n:>8.2f} {sum(r['score'] == 1 for r in rs):>4}/{n} "
            f"{sum(r['batches_read'] == r['n_batches'] for r in rs):>6}/{n} "
            f"{sum(bool(r['stats'].get('context_overflow')) for r in rs):>6}/{n} "
            f"{_mean_stat(rs, 'peak_context_tokens'):>9.0f} {_mean_stat(rs, 'prefill_tokens_with_prefix_reuse'):>23.0f} "
            f"{sum(r['prompt_tokens'] for r in rs) / n:>11.0f}"
        )
    errors = [r for r in rows if r["error"]]
    for r in errors:
        print(f"! {r['method']} seed {r['seed']}: {r['error']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=str)


if __name__ == "__main__":
    main()
