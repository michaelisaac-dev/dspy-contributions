# dspy.CLM

`CLM` (Context Language Model) is a DSPy module that lets the LM manage its own live context. It implements the zero-shot, context-as-a-file approach from ["Context Language Models" (Shao et al., 2026)](https://arxiv.org/abs/2609.37725).

`CLM` runs the same sandboxed REPL loop as [`RLM`](RLM.md), with one change: the trajectory is no longer append-only. Before every step the live context is written to a file whose path is bound to `CONTEXT_FILE` in the sandbox. Whatever that file holds after the step's code runs becomes the context for the next step. The LM can drop stale outputs, keep a tracker up to date in place, or compact finished work into a note, using ordinary Python.

| | Append-only loop (`RLM`, `ReAct`) | `CLM` |
|---|---|---|
| Next context | `c[t+1] = c[t] + step(c[t])` | `c[t+1] = f(c[t])`, chosen by the LM |
| Decides what to read in | RLM: yes | yes (same REPL, tools, `llm_query`) |
| Decides what to keep | no | yes |

## When to Use CLM

Use CLM when an agent runs for many steps and its context fills with material that matters only briefly: tool outputs, streamed input, search results, dead ends. RLM keeps huge *inputs* out of the prompt, but every step it takes is still appended to its history. CLM also controls that history, so it can work within a fixed `context_budget`.

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5-20251001"))

def next_batch() -> str:
    """Return the next batch of the stream."""
    ...

clm = dspy.CLM("task -> answer", tools=[next_batch], context_budget=2000, max_iters=30)
result = clm(task="Process every batch, then answer the question in the last one.")

print(result.answer)
print(result.context_stats)   # edits, peak context, prefix-reuse prefill tokens, overflow, ...
print(result.final_context)   # the live context as the LM left it
```

## How It Works

Each step:

1. **Meter.** If the live context is over `context_budget`, a warning turn is added. If it is still over budget at the next step, the run ends and the extract predictor answers from the newest turns that fit.
2. **Mirror.** The live context is written to `CONTEXT_FILE` inside the sandbox.
3. **Act.** The LM sees the instructions, the input metadata, and its live context, then writes code. That code can call tools and `llm_query`, and can rewrite `CONTEXT_FILE`.
4. **Read back.** If the file changed, it replaces the live context, with turn headers renumbered. With `edit_gate="fit"` (the default), an edit that grows the context past the budget is rejected; with `edit_gate="shrink"`, every edit that grows it is rejected. The step's code and output are then appended as `[[CTX_TURN i role=assistant]]` and `[[CTX_TURN i role=tool]]` blocks, followed by a `[context: ~N/B tokens]` readout.
5. **Reminders.** As in the paper's harness, a one-time reminder appears when the context first crosses 25%, 50% and 75% of the budget, re-armed after a compaction. Above 90%, an urgent reminder appears on every step.
6. **Free edits.** A step that only edits the context and prints nothing does not count against `max_iters`. These steps are capped by `max_edit_turns`.

Because the context is a single text field that DSPy renders fresh on every call, reading an edit back takes nothing more than replacing that text. No chat-message reconstruction is needed.

### Cost accounting

`context_stats["prefill_tokens_with_prefix_reuse"]` estimates what a prefix-caching server would prefill. On each call, it counts the managed context from the first character that differs from the previous call's context. Appending is cheap, and an edit near the top forces the whole tail to be re-read. This follows the paper's *prefix-reuse FLOPs*, restricted to the managed context. Token counts are approximate (~4 characters per token).

### Steering and customization

- `context_budget=None` drops the limit: the LM still gets size readouts and may edit, but nothing is enforced. In our runs, models rarely compact without a budget to stay under.
- `context_instructions="..."` replaces the built-in context-management instructions, for example with a skill document that prescribes a strategy (the paper's Section 4.2). `{context_budget}` is filled in.
- Subclasses can override `_reminder`, `_urgent_reminder`, and `_context_path`. The tutorials use these to run CLM with the paper's own prompt and reminder wording, loaded from a local clone of its repo.

## Example: CLM vs. RLM on a streamed ledger

[`tutorials/clm`](https://github.com/stanfordnlp/dspy/tree/main/docs/docs/tutorials/clm) has a small ContextBench-style demo. A ledger arrives batch by batch through a tool, mixed with distractors written in natural language, and the agent must report final balances. It compares stock `RLM`, `RLM` held to the same budget, and `CLM`.

## Output

`CLM` returns a `Prediction` with the signature's outputs plus:

- `trajectory`: every step's reasoning, code, output, and context-edit receipt, unedited
- `final_context`: the live context at the end of the run
- `context_stats`: `steps`, `free_edit_turns`, `edits_applied`, `edits_rejected`, `context_overflow`, `lm_calls`, `peak_context_tokens`, `total_context_tokens`, `prefill_tokens_with_prefix_reuse`, `final_context_tokens`
- `final_reasoning`

!!! note "Interpreter Requirements"
    Like RLM, CLM defaults to `PythonInterpreter` (Deno + Pyodide). The context mirror lives in the sandbox filesystem, so any `CodeInterpreter` whose code can read and write files works.

## API Reference

<!-- START_API_REF -->
::: dspy.CLM
    handler: python
    options:
        members:
            - __init__
            - __call__
            - forward
            - aforward
        show_source: true
        show_root_heading: true
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
