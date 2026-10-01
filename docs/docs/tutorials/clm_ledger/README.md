# CLM vs. RLM on a streamed ledger

A small demo of [`dspy.CLM`](../../api/modules/CLM.md), the DSPy implementation of
["Context Language Models"](https://arxiv.org/abs/2609.37725) (Shao et al., 2026). It contrasts CLM with `dspy.RLM`.

## The task

This is a ContextBench-style diagnostic: the input is a **stream** the agent pulls with a `next_batch()` tool, so whatever
it reads lands in its context. Each episode has 12 batches. Every batch has five ledger events (payments,
deposits, withdrawals, and corrections that cancel a payment from an *earlier* batch) mixed with distractors: planned
payments, declined requests, estimates, net-zero exchanges, and audit noise. The phrasing varies, and some amounts are
written out in words, so the agent has to read the events rather than regex them. A final batch asks for three of the
five balances, so the agent must track all five throughout.

Each episode is generated from a seed. The stream is ~3K tokens, about 1.5x a 2,000-token budget before any code
or reasoning is counted.

## Programs

All three use the same model, tool, signature, and `max_iters=30`:

| program | context |
|---|---|
| `RLM` | stock `dspy.RLM`; history is append-only, no budget |
| `RLM @ budget` | stock `dspy.RLM` loop, held to the 2,000-token budget with CLM's overflow rule |
| `CLM @ budget` | `dspy.CLM(context_budget=2000)`; the LM may rewrite its live context through `CONTEXT_FILE` |

The overflow rule: if the managed context is over budget at two consecutive steps, the run ends and the extract step
answers from the newest turns that fit.

## Run

```bash
pip install "dspy[deno]"
export ANTHROPIC_API_KEY=...
python docs/docs/tutorials/clm_ledger/clm_vs_rlm.py --model anthropic/claude-haiku-4-5-20251001 --seeds 8 --out results.json
```

## Results

`anthropic/claude-haiku-4-5-20251001`, 8 episodes (seeds 0-7), 2,000-token budget, temperature 0:

| program | accuracy | all 3 correct | read whole stream | overflowed | peak context | prefill w/ prefix reuse | prompt tokens / episode |
|---|---|---|---|---|---|---|---|
| `RLM` | 0.71 | 5/8 | 8/8 | 0/8 | ~12.0K | ~12.0K | ~98.8K |
| `RLM @ budget` | 0.00 | 0/8 | 0/8 | 8/8 | ~2.5K | ~2.5K | ~5.0K |
| `CLM @ budget` | 0.62 | 3/8 | 8/8 | 0/8 | ~2.7K | ~11.2K | ~39.5K |

*Accuracy* is the fraction of the three queried balances that are exactly right. *Peak context* and *prefill* cover
the managed context only (~4 chars/token). *Prompt tokens* are the provider-reported totals, including instructions.

What this shows:

- **Under a budget, context management is the difference between finishing and not.** Held to the same
  budget, RLM overflows after about four batches every time: it cannot remove anything from its history. CLM stays
  within budget on every episode by keeping a balance tracker in its context and periodically wiping the
  processed batches. It averages ~5 applied edits per episode and peaks around 2.7K tokens, briefly over budget
  before it compacts.
- **Against unbounded RLM, CLM uses ~2.5x fewer prompt tokens at similar accuracy.** 0.62 vs. 0.71 on 8
  episodes is about one episode's worth of difference. In both programs the errors are misread events (e.g. an
  amount off by $5-$50 in one batch), not lost state.
- **Edits are not free under prefix caching.** CLM re-sends far less context in total, but each compaction
  rewrites the top of the context, so its prefix-reuse prefill (~11.2K) is about the same as append-only RLM's
  (~12.0K). This is the cost the paper's Suffix Cache Reuse targets.

This is a small diagnostic, not a benchmark: one cheap model, 8 episodes, and no prompt optimization for either
program.
