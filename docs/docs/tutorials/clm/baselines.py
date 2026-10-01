"""Programs shared by the CLM demos: stock RLM, RLM held to a budget, and CLM, all metered the same way."""

from __future__ import annotations

from typing import Any, Callable

import dspy
from dspy.predict.clm import ContextMeter, approx_tokens
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory

METHODS = ("RLM", "CLM", "RLM @ budget", "CLM @ budget")


class BudgetedRLM(dspy.RLM):
    """Stock RLM held to a token budget on its history, with CLM's overflow rule.

    RLM cannot remove anything from its history, so the only change is the stop rule: a run whose
    history stays over budget for two consecutive steps ends and answers from the newest steps that fit.
    """

    def __init__(self, *args, context_budget: int, **kwargs):
        self.context_budget = context_budget
        super().__init__(*args, **kwargs)

    def forward(self, **input_args) -> Prediction:
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields)
        variables = self._build_variables(**input_args)
        meter, warned = ContextMeter(), False
        with self._interpreter_context(self._prepare_execution_tools(), None) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)
            for iteration in range(self.max_iters):
                text = history.format() if history else ""
                if approx_tokens(text) > self.context_budget:
                    if warned:
                        break
                    warned = True
                meter.observe(text)
                result = self._execute_iteration(repl, variables, history, iteration, regular_args, output_field_names)
                if isinstance(result, Prediction):
                    result.context_stats = {**meter.as_dict(), "context_overflow": False}
                    return result
                history = result
            overflow = approx_tokens(history.format()) > self.context_budget
            while len(history.entries) > 1 and approx_tokens(history.format()) > self.context_budget:
                history = REPLHistory(entries=history.entries[1:], max_output_chars=self.max_output_chars)
            prediction = self._extract_fallback(variables, history, output_field_names)
            prediction.context_stats = {**meter.as_dict(), "context_overflow": overflow}
            return prediction


def _with_step_hook(cls: type, hook: Callable[[], None] | None) -> type:
    """Subclass ``cls`` so ``hook`` runs before each step's code executes (RLM and CLM both run code there)."""
    if hook is None:
        return cls

    class Hooked(cls):
        def _execute_code(self, repl, code, input_args):
            hook()
            return super()._execute_code(repl, code, input_args)

    Hooked.__name__ = cls.__name__
    return Hooked


def run_program(
    method: str,
    signature: Any,
    tools: list[Callable],
    max_iters: int,
    budget: int,
    clm_class: type[dspy.CLM] | None = None,
    module_kwargs: dict[str, Any] | None = None,
    step_hook: Callable[[], None] | None = None,
    **inputs: Any,
) -> Prediction:
    """Run one of METHODS. Every prediction carries ``context_stats`` metered on the managed context.

    ``module_kwargs`` (e.g. ``max_output_chars``) are passed to every module alike, and ``step_hook`` runs
    before every step of every method (e.g. to reset a per-step tool-call limit).
    """
    kw = {"max_iters": max_iters, "tools": tools, **(module_kwargs or {})}
    if method == "RLM":
        # Stock RLM: meter its history from the trajectory afterwards (it is append-only, so this is exact).
        pred = _with_step_hook(dspy.RLM, step_hook)(signature, **kw)(**inputs)
        meter, history = ContextMeter(), REPLHistory()
        for entry in pred.trajectory:
            meter.observe(history.format() if history else "")
            history = history.append(**entry)
        pred.context_stats = {**meter.as_dict(), "context_overflow": False}
        return pred
    if method == "RLM @ budget":
        return _with_step_hook(BudgetedRLM, step_hook)(signature, context_budget=budget, **kw)(**inputs)
    if method == "CLM":
        return _with_step_hook(dspy.CLM, step_hook)(signature, context_budget=None, **kw)(**inputs)
    if method == "CLM @ budget":
        return _with_step_hook(clm_class or dspy.CLM, step_hook)(signature, context_budget=budget, **kw)(**inputs)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
