"""Tests for the CLM (Context Language Model) module.

- Unit tests: ContextFile, ContextMeter, signatures (no Deno)
- Integration tests (@pytest.mark.deno): DummyLM driving CLM in the PythonInterpreter sandbox
"""

import pytest

import dspy
from dspy.predict.clm import CLM, ContextFile, ContextMeter, approx_tokens
from dspy.utils.dummies import DummyLM

# ============================================================================
# Unit Tests
# ============================================================================


class TestContextFile:
    def test_append_numbers_turns(self):
        ctx = ContextFile().append_turn("assistant", "first").append_turn("tool", "out")
        assert ctx.text == "[[CTX_TURN 1 role=assistant]]\nfirst\n\n[[CTX_TURN 2 role=tool]]\nout"
        assert ctx.n_turns == 2

    def test_empty_context_has_placeholder(self):
        assert "empty" in ContextFile().format()

    def test_replaced_renumbers_surviving_headers(self):
        ctx = ContextFile().replaced("[[CTX_TURN 3 role=assistant]]\na\n\n[[CTX_TURN 7 role=notes]]\nb")
        assert ctx.text == "[[CTX_TURN 1 role=assistant]]\na\n\n[[CTX_TURN 2 role=notes]]\nb"

    def test_drop_oldest_turns_fits_budget(self):
        ctx = ContextFile()
        for i in range(10):
            ctx = ctx.append_turn("tool", f"output {i} " + "x" * 400)
        trimmed = ctx.drop_oldest_turns(budget=300)
        assert trimmed.tokens <= 300
        assert "output 9" in trimmed.text
        assert "output 0" not in trimmed.text
        assert trimmed.text.startswith("[[CTX_TURN 1 role=tool]]")


class TestContextMeter:
    def test_append_only_prefills_only_new_text(self):
        meter = ContextMeter()
        meter.observe("a" * 400)
        meter.observe("a" * 800)
        assert meter.prefill_tokens == approx_tokens("a" * 400) * 2
        assert meter.total_tokens == 300
        assert meter.peak_tokens == 200

    def test_edit_at_top_reprefills_tail(self):
        meter = ContextMeter()
        meter.observe("a" * 400 + "b" * 400)
        meter.observe("c" + "b" * 400)
        assert meter.prefill_tokens == 200 + approx_tokens("c" + "b" * 400)


class TestCLMSignatures:
    def test_action_signature_uses_live_context(self):
        clm = CLM("query -> answer", context_budget=1234)
        fields = clm.generate_action.signature.input_fields
        assert list(fields) == ["variables_info", "live_context", "iteration"]
        assert "repl_history" not in clm.extract.signature.input_fields
        assert "live_context" in clm.extract.signature.input_fields
        instructions = clm.generate_action.signature.instructions
        assert "CONTEXT_FILE" in instructions
        assert "1234 tokens" in instructions

    def test_no_budget_instructions(self):
        instructions = CLM("query -> answer", context_budget=None).generate_action.signature.instructions
        assert "no hard limit" in instructions
        assert "the run ends" not in instructions

    def test_context_file_name_is_reserved(self):
        with pytest.raises(ValueError, match="conflict"):
            CLM("CONTEXT_FILE -> answer")

    def test_result_names_are_reserved(self):
        with pytest.raises(ValueError, match="conflict"):
            CLM("query -> context_stats")


# ============================================================================
# Integration Tests
# ============================================================================

COMPACT_TURNS = """import re
s = open(CONTEXT_FILE).read()
s = re.sub(r"\\[\\[CTX_TURN 2 [^\\]]*\\]\\].*?(?=\\n\\[\\[CTX_TURN|\\Z)", "[[CTX_TURN 2 role=notes]]\\nbig output: first value is 0", s, flags=re.S)
open(CONTEXT_FILE, "w").write(s)"""


@pytest.mark.deno
class TestCLMWithDummyLM:
    def test_edit_replaces_live_context_and_is_free(self):
        lm = DummyLM([
            {"reasoning": "Look at a lot of data", "code": "print(list(range(400)))"},
            {"reasoning": "Collapse the big output", "code": COMPACT_TURNS},
            {"reasoning": "Done", "code": "SUBMIT(0)"},
        ])
        clm = CLM("query -> answer: int", max_iters=5, context_budget=5000)
        with dspy.context(lm=lm):
            result = clm(query="first value?")

        assert result.answer == 0
        stats = result.context_stats
        assert stats["edits_applied"] == 1
        assert stats["free_edit_turns"] == 1
        assert stats["steps"] == 2
        assert "big output: first value is 0" in result.final_context
        assert "399" not in result.final_context
        # The unedited trajectory keeps everything.
        assert "399" in result.trajectory[0]["output"]

        # The third call saw the compacted context, not the raw output.
        third_prompt = lm.history[2]["messages"][-1]["content"]
        assert "big output: first value is 0" in third_prompt
        assert "399" not in third_prompt

    def test_edit_that_matches_nothing_gets_receipt(self):
        lm = DummyLM([
            {"reasoning": "Try an edit", "code": "s = open(CONTEXT_FILE).read()\nopen(CONTEXT_FILE, 'w').write(s)"},
            {"reasoning": "Done", "code": "SUBMIT(1)"},
        ])
        clm = CLM("query -> answer: int", max_iters=5)
        with dspy.context(lm=lm):
            result = clm(query="q")
        assert result.context_stats["edits_applied"] == 0
        assert "NO change" in result.trajectory[0]["context_edit"]

    def test_overflow_ends_run_and_extracts(self):
        lm = DummyLM([
            {"reasoning": "Flood the context", "code": "print('y' * 4000)"},
            {"reasoning": "Ignore the warning", "code": "print('still here')"},
            {"answer": "7"},
        ])
        clm = CLM("query -> answer: int", max_iters=10, context_budget=500)
        with dspy.context(lm=lm):
            result = clm(query="q")
        assert result.answer == 7
        assert result.context_stats["context_overflow"] is True
        assert result.context_stats["steps"] == 2
        assert "OVER budget" in result.final_context

    def test_growing_edit_past_budget_is_rejected(self):
        lm = DummyLM([
            {"reasoning": "Bloat the context file", "code": "open(CONTEXT_FILE, 'a').write('z' * 4000)"},
            {"reasoning": "Done", "code": "SUBMIT(2)"},
        ])
        clm = CLM("query -> answer: int", max_iters=5, context_budget=500)
        with dspy.context(lm=lm):
            result = clm(query="q")
        assert result.answer == 2
        assert result.context_stats["edits_rejected"] == 1
        assert "zzzz" not in result.final_context

    def test_no_budget_never_overflows(self):
        lm = DummyLM([
            {"reasoning": "Flood the context", "code": "print('y' * 40000)"},
            {"reasoning": "Keep going", "code": "print('still here')"},
            {"reasoning": "Done", "code": "SUBMIT(3)"},
        ])
        clm = CLM("query -> answer: int", max_iters=5, context_budget=None)
        with dspy.context(lm=lm):
            result = clm(query="q")
        assert result.answer == 3
        assert result.context_stats["context_overflow"] is False
        assert "OVER budget" not in result.final_context
        assert "[context: ~" in result.final_context

    @pytest.mark.asyncio
    async def test_aforward(self):
        lm = DummyLM([
            {"reasoning": "Compute", "code": "print(6 * 7)"},
            {"reasoning": "Submit", "code": "SUBMIT(42)"},
        ])
        clm = CLM("query -> answer: int", max_iters=3)
        with dspy.context(lm=lm):
            result = await clm.aforward(query="q")
        assert result.answer == 42
        assert result.context_stats["lm_calls"] == 2
