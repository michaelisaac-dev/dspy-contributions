"""Run dspy.CLM with the CLM paper's own prompt and budget reminders, loaded from a local clone of its repo.

The paper's code and prompts are CC BY-NC 4.0, so none of that text ships with DSPy. Clone
https://github.com/facebookresearch/context-language-models and pass its path; this module reads the
"Managing your context" section of `clm/clm_harness/clm_agent/prompts.yaml` and imports the harness's
own `BudgetController` for the 25/50/75% and 90% reminder wording.

The only adaptations: the paper's agent edits the file from bash with a `python3 - <<'PY' ... PY`
heredoc, while CLM's code already runs in Python, so the heredoc wrapper lines are dropped from the
example; and the reminders are shown in the step's output instead of as separate user messages.
"""

from __future__ import annotations

import ast
import os
import re
import sys

import yaml

import dspy

PAPER_CONTEXT_FILE = "/tmp/.live_ctx/LIVE_CTX_MAIN.txt"


def _context_section(repo: str) -> str:
    with open(os.path.join(repo, "clm/clm_harness/clm_agent/prompts.yaml")) as f:
        template = yaml.safe_load(f)["system_template"]
    section = template[template.index("## Managing your context") : template.index("{{finish_instructions}}")]
    section = "\n".join(line for line in section.splitlines() if line.strip() not in ("python3 - <<'PY'", "PY"))
    return section.replace("{{context_budget}}", "{context_budget}").strip()


def _compaction_hint(repo: str) -> str:
    source = open(os.path.join(repo, "clm/clm_harness/clm_agent/harness.py")).read()
    expression = re.search(r"\n\s*hint = (\(.*?\n\s*\))\n", source, re.S).group(1)
    return eval(compile(ast.parse(expression, mode="eval"), "hint", "eval"), {"_CTX_FILE": PAPER_CONTEXT_FILE})


def paper_clm_class(repo: str) -> type[dspy.CLM]:
    """A CLM subclass that uses the paper's context prompt, reminders, file path, and shrink edit gate."""
    sys.path.insert(0, os.path.join(repo, "clm"))
    from clm_harness.utils.budget import BudgetController

    section, hint = _context_section(repo), _compaction_hint(repo)

    class PaperCLM(dspy.CLM):
        def __init__(self, *args, context_budget: int, **kwargs):
            kwargs.setdefault("edit_gate", "shrink")  # the paper's BrowseComp-Plus config
            super().__init__(*args, context_budget=context_budget, context_instructions=section, **kwargs)
            self._paper_budget = BudgetController(
                context_budget,
                reserve_tokens=0,
                nudge_ratios=list(self.REMINDER_LEVELS),
                persistent_nudge_ratio=self.URGENT_REMINDER_LEVEL,
                compaction_hint=hint,
            )

        def _context_path(self) -> str:
            # Each Pyodide sandbox has its own in-memory filesystem, so the paper's fixed path is safe.
            return PAPER_CONTEXT_FILE

        def _reminder(self, level: float, tokens: int) -> str:
            return self._paper_budget.nudge_message(level, tokens)

        def _urgent_reminder(self, tokens: int) -> str:
            return self._paper_budget.persistent_nudge_message(tokens)

    return PaperCLM
