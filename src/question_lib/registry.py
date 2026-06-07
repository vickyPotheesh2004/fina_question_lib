"""
question_lib/registry.py
Single API surface. One import gets you everything.

    from question_lib import (
        parse_question,      # Question text → QuestionPlan
        execute_plan,        # QuestionPlan + cells/raw_text → ExecutionResult
        answer_question,     # one-shot: question text → final answer
    )
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .decomposer     import parse_question
from .plan_executor  import execute_plan
from .models import (
    ExecutionResult, ExtractRequest, Intent, Modifier, ModifierKind,
    Operation, Period, Polarity, QuestionPlan, Subject, SubFormula,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# One-shot helper
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    cells:    Optional[List[Dict]] = None,
    raw_text: str = "",
    company:  str = "",
    fiscal_year: str = "",
    doc_type: str = "",
) -> ExecutionResult:
    """
    One-shot entry point: parse + execute in a single call.

    Returns ExecutionResult.answered = True if we got a final answer.
    Returns ExecutionResult.answered = False (and reasons in audit_trail)
    when the question couldn't be answered deterministically.
    """
    plan = parse_question(question)
    return execute_plan(
        plan=plan,
        cells=cells or [],
        raw_text=raw_text or "",
        company=company,
        fy=fiscal_year,
        doc_type=doc_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def lib_status() -> Dict[str, bool]:
    """Return which optional libs are available."""
    out = {}
    for lib in ("extract_lib", "maths_lib", "logic_lib", "format_lib", "verify_lib"):
        try:
            __import__(lib)
            out[lib] = True
        except Exception:
            out[lib] = False
    return out


def describe_plan(plan: QuestionPlan) -> str:
    """Human-readable summary of a parsed plan."""
    lines = [
        f"intent     = {plan.intent.value}",
        f"polarity   = {plan.polarity.value}",
        f"operation  = {plan.operation.value}",
        f"subject    = {plan.subject.metric_id if plan.subject else None}",
        f"periods    = {[p.fiscal_year for p in plan.periods]}",
        f"modifiers  = {[m.kind for m in plan.modifiers]}",
        f"sub_forms  = {[s.name for s in plan.sub_formulas]}",
        f"extracts   = {[(e.metric_id, e.period) for e in plan.required_extracts]}",
        f"output     = {plan.output_metric} ({plan.output_unit})",
        f"rule       = {plan.decision_rule_id}",
        f"confidence = {plan.confidence:.2f}",
    ]
    return "\n  ".join(lines)
