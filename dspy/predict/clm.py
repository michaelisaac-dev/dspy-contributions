"""
Context Language Model (CLM) module for DSPy.

A CLM manages its own live context. The context is treated as a file: before every step it is
mirrored into the sandbox, the LM may rewrite that file with ordinary code, and whatever the file
holds afterwards becomes the context for the next step. A standard agent loop only appends to its
context (``c_{t+1} = c_t + f(c_t)``); a CLM produces the next context itself (``c_{t+1} = f(c_t)``),
so it can drop stale outputs, keep trackers up to date in place, or compact finished work.

This module is RLM's REPL loop with the append-only history replaced by an editable context file
under a token budget. RLM decides *what to read* into the context; CLM also decides *what to keep*.

Reference: "Context Language Models" (Shao et al., 2026), arXiv:2609.37725
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import pydantic

import dspy
from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLVariable
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import format_error_for_lm

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

__all__ = ["CLM", "ContextFile", "ContextMeter", "approx_tokens"]

CONTEXT_INSTRUCTIONS_TEMPLATE = r"""

Managing your live context:
- `live_context` is your working memory. Each step (your reasoning and code, then its output) is appended to it as `[[CTX_TURN i role=...]]` blocks. Apart from the inputs and your REPL variables, it is all you will see next step.
- Before every step, the live context is mirrored to the file at path `CONTEXT_FILE`. Whatever that file holds after your code runs REPLACES your live context. Use it to delete stale outputs, collapse finished work into short notes, or keep a tracker up to date in place.
{budget_rule}
- Edit with code; never retype text you have already seen:
    s = open(CONTEXT_FILE).read()
    s = re.sub(r"(\[\[CTX_TURN 3 [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)", r"\1\n[turn 3 done: ids are in column 2]", s, flags=re.S)
    with open(CONTEXT_FILE, "w") as f: f.write(s)
- Keep the header line of any turn you keep; headers are renumbered 1..k after each edit. Text outside a turn survives as a note.
- A step that edits the context and prints nothing is free: it does not count against your iterations. Do not print the file; it is already in front of you.
- Compact in batches: one larger edit beats many tiny ones, because everything after an edited spot has to be re-read. Write summaries that keep exactly what you will still need (values, ids, decisions, what remains)."""

_BUDGET_RULE = (
    "- Budget: keep the live context under {context_budget} tokens. Every output ends with a "
    "`[context: ~N/{context_budget} tokens]` readout. If you go over budget and do not compact on your very next "
    "step, the run ends."
)
_NO_BUDGET_RULE = (
    "- Size: every output ends with a `[context: ~N tokens]` readout. There is no hard limit, but a long context "
    "is slower, costlier, and harder to reason over, so compact it as it grows."
)

_HEADER_RE = re.compile(r"^\[\[CTX_TURN\s+\d+\s+role=([A-Za-z_]+)\]\][ \t]*$", re.M)

# Harness code run around each step. Private names keep the model's namespace clean, and
# _CLM_PATH stays correct even if the model rebinds CONTEXT_FILE.
_WRITE_MIRROR = """\
import os as _clm_os
_clm_os.makedirs(_clm_os.path.dirname(_CLM_PATH), exist_ok=True)
with open(_CLM_PATH, "w") as _clm_f:
    _clm_f.write(_clm_text)
del _clm_os, _clm_f, _clm_text
"""
_READ_MIRROR = """\
import json as _clm_json
try:
    with open(_CLM_PATH) as _clm_f:
        print(_clm_json.dumps(_clm_f.read()))
    del _clm_f
except FileNotFoundError:
    print(_clm_json.dumps(None))
"""
_REMOVE_MIRROR = """\
import os as _clm_os, shutil as _clm_shutil
_clm_shutil.rmtree(_clm_os.path.dirname(_CLM_PATH), ignore_errors=True)
del _clm_os, _clm_shutil
"""


def approx_tokens(text: str) -> int:
    """Model-agnostic token estimate (~4 characters per token)."""
    return (len(text) + 3) // 4


def _renumber(text: str) -> str:
    counter = iter(range(1, 1_000_000))
    return _HEADER_RE.sub(lambda m: f"[[CTX_TURN {next(counter)} role={m.group(1)}]]", text)


def _is_echoed_write_count(code: str, output: str) -> bool:
    """True when the only output is the REPL echoing a trailing ``f.write(...)`` call's return value."""
    lines = code.strip().splitlines()
    return output.strip().isdigit() and bool(lines) and ".write(" in lines[-1]


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class ContextFile(pydantic.BaseModel):
    """The CLM's live context: free text made of ``[[CTX_TURN i role=...]]`` blocks.

    Immutable: ``append_turn`` and ``replaced`` return new instances.
    """

    text: str = ""

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def tokens(self) -> int:
        return approx_tokens(self.text)

    @property
    def n_turns(self) -> int:
        return len(_HEADER_RE.findall(self.text))

    def format(self) -> str:
        return self.text if self.text.strip() else "Your live context is empty: you have not taken any steps yet."

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append_turn(self, role: str, body: str) -> ContextFile:
        block = f"[[CTX_TURN {self.n_turns + 1} role={role}]]\n{body.strip()}"
        return ContextFile(text=f"{self.text.rstrip()}\n\n{block}" if self.text.strip() else block)

    def replaced(self, text: str) -> ContextFile:
        return ContextFile(text=_renumber(text.strip()))

    def drop_oldest_turns(self, budget: int) -> ContextFile:
        """Drop whole turns from the top until the context fits ``budget`` tokens."""
        text = self.text
        while approx_tokens(text) > budget:
            headers = list(_HEADER_RE.finditer(text))
            if len(headers) < 2:
                return ContextFile(text=text[-budget * 4:])
            text = text[headers[1].start():]
        return self.replaced(text)


@dataclass
class ContextMeter:
    """Cost accounting for a context that is re-sent on every LM call.

    ``prefill_tokens`` is what a server with prefix caching must prefill: on each call, everything from
    the first character that differs from the previous call's context onward. Appending costs only the
    new text; an edit near the top forces the whole tail to be re-read. This mirrors the paper's
    prefix-reuse FLOPs, restricted to the managed context (instructions and inputs are a fixed prefix).
    """

    calls: int = 0
    peak_tokens: int = 0
    total_tokens: int = 0
    prefill_tokens: int = 0
    _previous: str = field(default="", repr=False)

    def observe(self, text: str) -> None:
        tokens = approx_tokens(text)
        self.calls += 1
        self.peak_tokens = max(self.peak_tokens, tokens)
        self.total_tokens += tokens
        self.prefill_tokens += approx_tokens(text[_common_prefix_len(text, self._previous):])
        self._previous = text

    def as_dict(self) -> dict[str, int]:
        return {
            "lm_calls": self.calls,
            "peak_context_tokens": self.peak_tokens,
            "total_context_tokens": self.total_tokens,
            "prefill_tokens_with_prefix_reuse": self.prefill_tokens,
        }


@dataclass
class _Run:
    """Mutable state for one CLM invocation."""

    path: str
    context: ContextFile = field(default_factory=ContextFile)
    meter: ContextMeter = field(default_factory=ContextMeter)
    steps: int = 0
    edit_turns: int = 0
    edits_applied: int = 0
    edits_rejected: int = 0
    over_budget_warned: bool = False
    reminders_fired: set[float] = field(default_factory=set)
    overflowed: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "free_edit_turns": self.edit_turns,
            "edits_applied": self.edits_applied,
            "edits_rejected": self.edits_rejected,
            "final_context_tokens": self.context.tokens,
            "context_overflow": self.overflowed,
            **self.meter.as_dict(),
        }


@experimental
class CLM(RLM):
    """Context Language Model module.

    Runs RLM's sandboxed REPL loop, but the trajectory is a live context the LM manages itself. Before
    each step the context is written to a file whose path is bound to ``CONTEXT_FILE`` in the sandbox;
    any change the step's code makes to that file becomes the next context. Each output carries a
    ``[context: ~N/B tokens]`` readout, and a run that stays over ``context_budget`` for two
    consecutive steps ends, answering from the newest turns that fit.

    Examples:
        ```python
        clm = dspy.CLM("task -> answer", tools=[next_batch], context_budget=3000, max_iters=30)
        result = clm(task="Process every batch, then answer the question at the end.")
        print(result.answer, result.context_stats)
        ```
    """

    _RESERVED_SANDBOX_NAMES = RLM._RESERVED_SANDBOX_NAMES | {"CONTEXT_FILE"}
    # Budget fractions that trigger a one-time reminder, and the level above which every step gets an urgent one.
    REMINDER_LEVELS = (0.25, 0.5, 0.75)
    URGENT_REMINDER_LEVEL = 0.9
    _RESERVED_RESULT_NAMES = RLM._RESERVED_RESULT_NAMES | {"final_context", "context_stats"}

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iters: int = 20,
        context_budget: int | None = 8_000,
        max_edit_turns: int | None = None,
        context_instructions: str | None = None,
        edit_gate: Literal["fit", "shrink"] = "fit",
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    ):
        """
        Args:
            signature: Defines inputs and outputs, as for RLM.
            max_iters: Maximum task steps. Edit-only steps that print nothing do not count.
            context_budget: Token budget (~4 chars/token) for the live context, or None for no limit (the LM
                still sees size readouts and may edit, but nothing is enforced). The instructions and
                input metadata sit outside it, like a pinned system prompt.
            max_edit_turns: Cap on free edit-only steps. Defaults to ``max_iters``.
            context_instructions: Replaces the built-in context-management instructions, e.g. to steer the
                strategy with a skill document. ``{context_budget}`` is filled in.
            edit_gate: ``"fit"`` accepts an edit that grows the context while it stays within budget;
                ``"shrink"`` rejects every edit that grows it.
            max_llm_calls, max_output_chars, verbose, tools, sub_lm, interpreter_factory: As for RLM.
        """
        self.context_budget = context_budget
        self.max_edit_turns = max_iters if max_edit_turns is None else max_edit_turns
        if edit_gate not in ("fit", "shrink"):
            raise ValueError(f"edit_gate must be 'fit' or 'shrink', not {edit_gate!r}")
        self.edit_gate = edit_gate
        self.context_instructions = context_instructions
        super().__init__(
            signature,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            sub_lm=sub_lm,
            interpreter_factory=interpreter_factory,
        )

    # =========================================================================
    # Signatures
    # =========================================================================

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """RLM's signatures with the append-only history swapped for the editable live context."""
        action_sig, extract_sig = super()._build_signatures()
        context_field = dspy.InputField(desc="Your live context, which you manage through CONTEXT_FILE")
        if self.context_instructions is not None:
            context_section = "\n\n" + self.context_instructions.replace("{context_budget}", str(self.context_budget))
        else:
            budget_rule = _NO_BUDGET_RULE if self.context_budget is None else _BUDGET_RULE.replace(
                "{context_budget}", str(self.context_budget)
            )
            context_section = CONTEXT_INSTRUCTIONS_TEMPLATE.replace("{budget_rule}", budget_rule)
        instructions = action_sig.instructions + context_section
        action_sig = (
            action_sig.delete("repl_history")
            .insert(1, "live_context", context_field, type_=ContextFile)
            .with_instructions(instructions)
        )
        extract_sig = (
            extract_sig.delete("repl_history")
            .insert(1, "live_context", dspy.InputField(desc="Your live context at the end of the run"), type_=ContextFile)
            .with_instructions(extract_sig.instructions.replace("REPL trajectory", "live context"))
        )
        return action_sig, extract_sig

    # =========================================================================
    # One step
    # =========================================================================

    def _context_path(self) -> str:
        """Where the live context is mirrored in the sandbox. Unique per run, for interpreters that share a disk."""
        return f"/tmp/.live_ctx/{uuid.uuid4().hex[:12]}/LIVE_CTX_MAIN.txt"

    def _start_run(self) -> _Run:
        return _Run(path=self._context_path())

    def _before_action(self, run: _Run) -> bool:
        """Meter the context the LM is about to see. Returns False when the run must end on overflow."""
        if self.context_budget is not None and run.context.tokens > self.context_budget:
            if run.over_budget_warned:
                run.overflowed = True
                return False
            run.over_budget_warned = True
            run.context = run.context.append_turn(
                "user",
                f"[harness] Your live context is OVER budget (~{run.context.tokens}/{self.context_budget} tokens). "
                "Compact CONTEXT_FILE below budget on this step, or the run ends.",
            )
        else:
            run.over_budget_warned = False
        run.meter.observe(run.context.text)
        return True

    def _action_inputs(self, variables: list[REPLVariable], run: _Run) -> dict[str, Any]:
        return {
            "signature": self._action_signature_for_current_factory(),
            "variables_info": [variable.format() for variable in variables],
            "live_context": run.context,
            "iteration": f"{run.steps + 1}/{self.max_iters}",
        }

    def _mirror(self, repl: CodeInterpreter, code: str, variables: dict[str, Any]) -> Any:
        try:
            return repl.execute(code, variables=variables)
        except (CodeExecutionError, SyntaxError) as e:
            logger.debug("CLM context mirror failed: %s", e)
            return None

    def _read_back(self, repl: CodeInterpreter, run: _Run) -> str | None:
        raw = self._mirror(repl, _READ_MIRROR, {"_CLM_PATH": run.path})
        if isinstance(raw, list):
            raw = "\n".join(map(str, raw))
        try:
            return json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            return None

    def _reconcile(self, run: _Run, written: str, edited: str | None, code: str) -> tuple[str, bool]:
        """Adopt the model's edit of the context file. Returns (receipt, edit_applied)."""
        if edited is None or edited.strip() == written.strip():
            names_file = "CONTEXT_FILE" in code or run.path.rsplit("/", 1)[-1] in code
            if names_file and "write" in code:
                return (
                    f"[CONTEXT_FILE: NO change - your edit matched nothing; context is still ~{run.context.tokens} tokens. "
                    "Match text you have seen, or target turn headers like `[[CTX_TURN 4 role=tool]]`.]"
                ), False
            return "", False

        before, after = run.context.tokens, approx_tokens(edited)
        over_budget = self.context_budget is not None and after > self.context_budget
        if after > before and (self.edit_gate == "shrink" or over_budget):
            run.edits_rejected += 1
            rule = "an edit must SHRINK the context" if self.edit_gate == "shrink" else (
                f"an edit must fit the {self.context_budget}-token budget"
            )
            return (
                f"[CONTEXT_FILE: edit REJECTED - it grew the context ~{before}->{after} tokens, and {rule}. "
                "Replace stale text with a SHORTER summary.]"
            ), False

        run.context = run.context.replaced(edited)
        run.edits_applied += 1
        return f"[CONTEXT_FILE: edit applied - context ~{before}->{run.context.tokens} tokens, {run.context.n_turns} turns]", True

    def _readout(self, run: _Run) -> str:
        tokens = run.context.tokens
        if self.context_budget is None:
            return f"[context: ~{tokens} tokens]"
        readout = f"[context: ~{tokens}/{self.context_budget} tokens]"
        fraction = tokens / self.context_budget
        # A compaction re-arms the reminders for the levels it dropped back below.
        run.reminders_fired = {level for level in run.reminders_fired if fraction >= level}
        if fraction > 1:
            return readout + "\n[OVER budget: compact CONTEXT_FILE on your next step, or the run ends.]"
        if fraction >= self.URGENT_REMINDER_LEVEL:
            return f"{readout}\n{self._urgent_reminder(tokens)}"
        crossed = [level for level in self.REMINDER_LEVELS if fraction >= level and level not in run.reminders_fired]
        if crossed:
            run.reminders_fired.update(crossed)
            readout += f"\n{self._reminder(max(crossed), tokens)}"
        return readout

    def _reminder(self, level: float, tokens: int) -> str:
        """One-time reminder when the context first crosses ``level`` of the budget."""
        return (
            f"[Reminder: your live context is at {round(100 * tokens / self.context_budget)}% of its budget. Compact "
            "stale turns into a short note when it is worth it; one larger compaction beats many small ones.]"
        )

    def _urgent_reminder(self, tokens: int) -> str:
        """Reminder shown on every step once the context is above ``URGENT_REMINDER_LEVEL`` of the budget."""
        return (
            f"[URGENT: only ~{self.context_budget - tokens} tokens of room left, which may not fit another output. "
            "Compact CONTEXT_FILE now.]"
        )

    def _run_action(
        self,
        repl: CodeInterpreter,
        action: Prediction,
        input_args: dict[str, Any],
        run: _Run,
        output_field_names: list[str],
    ) -> Prediction | None:
        """Execute one step against the mirrored context. Returns a Prediction once SUBMIT succeeds."""
        if self.verbose:
            logger.info(f"CLM step {run.steps + 1}/{self.max_iters}\nReasoning: {action.reasoning}\nCode:\n{action.code}")

        written = run.context.text
        self._mirror(repl, _WRITE_MIRROR, {"_CLM_PATH": run.path, "_clm_text": written})
        try:
            code = _strip_code_fences(action.code)
            result = self._execute_code(repl, code, {**input_args, "CONTEXT_FILE": run.path})
        except SyntaxError as e:
            code, result = action.code, f"[Error] {format_error_for_lm(e)}"

        receipt, edited = self._reconcile(run, written, self._read_back(repl, run), code)

        if isinstance(result, FinalOutput):
            parsed, error = self._process_final_output(result, output_field_names)
            if error is None:
                run.steps += 1
                run.log.append({"reasoning": action.reasoning, "code": code, "output": f"FINAL: {parsed}"})
                return Prediction(
                    **parsed,
                    trajectory=run.log,
                    final_reasoning=action.reasoning,
                    final_context=run.context.text,
                    context_stats=run.stats(),
                )
            result = error

        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""
        is_error = output.startswith(("[Error]", "[Type Error]"))
        printed_nothing = not output or _is_echoed_write_count(code, output)

        if edited and printed_nothing and not is_error and run.edit_turns < self.max_edit_turns:
            run.edit_turns += 1
        else:
            run.steps += 1
        run.log.append({"reasoning": action.reasoning, "code": code, "output": output, "context_edit": receipt})

        shown = REPLEntry.format_output(output, self.max_output_chars) if output else "(no output)"
        run.context = run.context.append_turn("assistant", f"Reasoning: {action.reasoning}\nCode:\n```python\n{code}\n```")
        run.context = run.context.append_turn("tool", "\n".join(filter(None, [shown, receipt])))
        run.context = ContextFile(text=f"{run.context.text}\n{self._readout(run)}")
        return None

    def _final_without_submit(self, run: _Run, extract_pred: Prediction, output_field_names: list[str]) -> Prediction:
        return Prediction(
            trajectory=run.log,
            final_reasoning="Context overflow forced final output" if run.overflowed else "Extract forced final output",
            final_context=run.context.text,
            context_stats=run.stats(),
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def _extract_inputs(self, variables: list[REPLVariable], run: _Run) -> dict[str, Any]:
        reason = "overflowed its context budget" if run.overflowed else "reached max iterations"
        logger.warning(f"CLM {reason}, using extract to get final output")
        return {
            "variables_info": [variable.format() for variable in variables],
            "live_context": run.context if self.context_budget is None else run.context.drop_oldest_turns(self.context_budget),
        }

    def _keep_going(self, run: _Run) -> bool:
        return run.steps < self.max_iters

    # =========================================================================
    # Public Interface
    # =========================================================================

    def forward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Execute CLM to produce outputs from the given inputs.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally, as for RLM.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with the signature's outputs, plus ``trajectory`` (every step, unedited),
            ``final_context`` (the live context as the LM left it), and ``context_stats``.
        """
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            run = self._start_run()
            try:
                while self._keep_going(run) and self._before_action(run):
                    action = self.generate_action(**self._action_inputs(variables, run))
                    prediction = self._run_action(repl, action, regular_args, run, output_field_names)
                    if prediction is not None:
                        return prediction
                extract_pred = self.extract(**self._extract_inputs(variables, run))
                return self._final_without_submit(run, extract_pred, output_field_names)
            finally:
                self._cleanup(repl, run)

    async def aforward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Async version of forward()."""
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            run = self._start_run()
            try:
                while self._keep_going(run) and self._before_action(run):
                    action = await self.generate_action.acall(**self._action_inputs(variables, run))
                    prediction = self._run_action(repl, action, regular_args, run, output_field_names)
                    if prediction is not None:
                        return prediction
                extract_pred = await self.extract.acall(**self._extract_inputs(variables, run))
                return self._final_without_submit(run, extract_pred, output_field_names)
            finally:
                self._cleanup(repl, run)

    def _cleanup(self, repl: CodeInterpreter, run: _Run) -> None:
        """Remove the mirror directory, which matters for interpreters backed by the real filesystem."""
        try:
            self._mirror(repl, _REMOVE_MIRROR, {"_CLM_PATH": run.path})
        except CodeInterpreterError:
            logger.debug("CLM could not remove its context mirror", exc_info=True)
