# CLM vs. RLM

Two comparisons of [`dspy.CLM`](../../api/modules/CLM.md), the DSPy implementation of
["Context Language Models"](https://arxiv.org/abs/2609.37725) (Shao et al., 2026), against `dspy.RLM`:

1. [A streamed ledger](#1-a-streamed-ledger): a small synthetic diagnostic (`ledger.py`).
2. [BrowseComp-Plus](#2-browsecomp-plus): the CLM paper's headline deep-research benchmark (`browsecomp.py`).

Files: `baselines.py` holds the shared programs (stock RLM, RLM held to a budget, CLM). `paper_prompt.py` runs CLM
with the paper's own prompt and reminders, read from a local clone of its repo; that text is CC BY-NC 4.0, so it is
not copied here.

# 1. A streamed ledger

### The task

This is a ContextBench-style diagnostic: the input is a **stream** the agent pulls with a `next_batch()` tool, so whatever
it reads lands in its context. Each episode has 12 batches. Every batch has five ledger events (payments,
deposits, withdrawals, and corrections that cancel a payment from an *earlier* batch) mixed with distractors: planned
payments, declined requests, estimates, net-zero exchanges, and audit noise. The phrasing varies, and some amounts are
written out in words, so the agent has to read the events rather than regex them. A final batch asks for three of the
five balances, so the agent must track all five throughout.

Each episode is generated from a seed. The stream is ~3K tokens, about 1.5x a 2,000-token budget before any code
or reasoning is counted.

### Programs

All four use the same model, tool, signature, and `max_iters=30`:

| program | context |
|---|---|
| `RLM` | stock `dspy.RLM`; history is append-only, no budget |
| `CLM` | `dspy.CLM(context_budget=None)`; identical to `RLM` except the LM can rewrite its live context (size readouts, no limit) |
| `RLM @ budget` | stock `dspy.RLM` loop, held to the 2,000-token budget with CLM's overflow rule |
| `CLM @ budget` | `dspy.CLM(context_budget=2000)`; the LM may rewrite its live context through `CONTEXT_FILE` |

The overflow rule: if the managed context is over budget at two consecutive steps, the run ends and the extract step
answers from the newest turns that fit.

### Run

```bash
pip install "dspy[deno]"
export ANTHROPIC_API_KEY=...
python docs/docs/tutorials/clm/ledger.py --model anthropic/claude-haiku-4-5-20251001 --seeds 8 --out results.json
```

### Results

`anthropic/claude-haiku-4-5-20251001`, 8 episodes (seeds 0-7), 2,000-token budget, temperature 0:

| program | accuracy | all 3 correct | read whole stream | overflowed | peak context | prefill w/ prefix reuse | prompt tokens / episode |
|---|---|---|---|---|---|---|---|
| `RLM` | 0.71 | 5/8 | 8/8 | 0/8 | ~12.0K | ~12.0K | ~98.8K |
| `CLM` (no budget) | 0.92 | 6/8 | 8/8 | 0/8 | ~11.9K | ~11.9K | ~134.3K |
| `RLM @ budget` | 0.00 | 0/8 | 0/8 | 8/8 | ~2.5K | ~2.5K | ~5.0K |
| `CLM @ budget` | 0.62 | 3/8 | 8/8 | 0/8 | ~2.7K | ~11.2K | ~39.5K |

*Accuracy* is the fraction of the three queried balances that are exactly right. *Peak context* and *prefill* cover
the managed context only (~4 chars/token). *Prompt tokens* are the provider-reported totals, including instructions.

What this shows:

- **Without a budget, CLM does not manage its context at all.** No-budget CLM applied **zero** edits in all 8
  episodes: with no limit and no pressure, the model just lets the context grow, so its peak (~11.9K) matches RLM's.
  Its higher accuracy (0.92 vs. 0.71) therefore cannot come from context management. What differs is the prompt
  and history format: the context-management instructions, `[[CTX_TURN]]` blocks, and size readouts. On 8
  episodes that gap is also within noise. Those extra instructions and readouts cost ~35% more prompt tokens than
  RLM. In this setup, CLM's mechanism only switches on once there is a budget to stay under. That matches the
  paper, where CLM always runs with a budget and an editing reminder near the limit.

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

# 2. BrowseComp-Plus

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) (830 hard fact-seeking questions over a fixed
~100K-document corpus) is the CLM paper's headline benchmark, and the paper includes RLM as a baseline. Here the
agent gets `search(query)`, which returns the top 5 documents with 512-token snippets via BM25 (`bm25s`, Lucene
parameters), and `get_document(docid)`, capped at 8,192 tokens. Each step's output is capped at 60,000 characters,
as in the paper's config. Answers are graded with the official BrowseComp-Plus grader template.

Methods: stock `RLM` (no budget), `RLM @ budget` (held to the budget with CLM's overflow rule), and
`CLM-paper @ budget`, which is CLM with the **paper's own prompt**, its 25/50/75/90% reminder wording, and the
"shrink" edit gate from its BrowseComp-Plus config. All three get the same instruction, which adds one line about
persistence ("every question has an answer in the corpus: do not give up"), because the agents otherwise quit
after a few steps.

```bash
pip install "dspy[deno]" datasets bm25s PyStemmer
git clone https://github.com/facebookresearch/context-language-models
python docs/docs/tutorials/clm/browsecomp_prepare.py --data-dir data/bcp     # ~3.3 GB corpus + BM25 index
python docs/docs/tutorials/clm/browsecomp.py --data-dir data/bcp --paper-repo context-language-models \
    --model openai/gpt-5.6-luna --judge-model openai/gpt-5.6-terra --n 30
```

The script reads `AZURE_BASE` (an OpenAI-compatible `/openai/v1` endpoint) and `AZURE_API_KEY`. `--calls-per-step 1`
limits the agent to one search or document read per step, like the paper's one-command-per-turn agent.

### Results: `gpt-5.6-luna`, 30 questions, batched searches allowed

24K-token budget, up to 40 steps, judged by `gpt-5.6-terra`, sample seed 0:

| method | accuracy | overflowed | mean steps | gave up ("unable to determine") | peak context | prompt tokens / q | $ / q |
|---|---|---|---|---|---|---|---|
| `RLM` | 0.43 (13/30) | 0/30 | 7.4 | 5 | ~15K | ~342K | $0.082 |
| `RLM @ budget` | 0.37 (11/30) | 18/30 | 3.9 | 7 | ~27K | ~49K | $0.014 |
| `CLM-paper @ budget` | 0.33 (10/30) | 4/30 | 8.2 | 11 | ~25K | ~147K | $0.042 |

- **No accuracy difference we can distinguish from noise.** Comparing question by question, RLM alone solved
  5 questions and CLM alone solved 2 (both solved 8; p ≈ 0.45). Only 15 of the 30 were solved by any method.
- **CLM does manage its context** (~2.5 edits per question, overflow 4/30 vs. 18/30 for budgeted RLM), and it uses
  2.3x fewer prompt tokens than unbudgeted RLM. Its edits rewrite the top of the context, though, so its
  prefix-reuse prefill (~68K) is the highest of the three.
- **Why the paper's gain doesn't show here:**
  - Runs are short (median 5-8 steps vs. up to 100 turns in the paper).
  - CLM gives up more often (11 vs. 5), after compacting away material it needed.
  - DSPy's RLM already reads selectively: it runs many searches in one step and slices documents with code.
  - A single step of batched searches can print ~15K tokens and blow the budget before CLM can compact.

### Pilot: `gpt-5.6-terra`, one search per step, 3 questions

Same setup with `--calls-per-step 1 --max-iters 60`. This is only a pilot: 3 questions, $14 total.

| question | `RLM` | `RLM @ budget` | `CLM-paper @ budget` |
|---|---|---|---|
| q624 | ✗ 5 steps, $0.08 | ✗ 3 steps, $0.03 | ✗ 5 steps, $0.07 |
| q403 | ✓ 6 steps, $0.14 | ✓ 10 steps, $0.35 | ✗ 27 steps, $0.98 |
| q1185 | ✗ 52 steps, peak ~89K, **$9.81** | ✗ overflow at 13 steps, $1.04 | ✗ 30 steps, peak ~20K, $1.64 |

- **q624** is a rounding miss shared by all three (65.5% reported as "66%" against a gold answer of "65%").
- **q403** is a real CLM miss: it followed a wrong lead for 21 searches.
- **q1185** shows the cost side. All three failed, but unbudgeted RLM's context grew to ~89K tokens and the run cost 6x
  more than CLM's, which compacted 3 times and stayed near 20K.
- With one call per step, runs do reach the long horizons the paper studies (27-52 steps on hard questions).
- Every agent followed the one-call rule (0 refused calls).

**Not yet run:** the 30-question terra comparison. Equal budgets for both methods, as in the paper, is the
recommended next step (~$30-50); a version that includes unbudgeted RLM costs ~$150-250, because RLM's context
grows without bound.
