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

try:
    from . import advanced_formulas as _adv
    from . import table_normalizer as _tnorm
    _HAS_ADV = True
except Exception:
    _adv = None
    _tnorm = None
    _HAS_ADV = False

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
    structured_tables: Optional[List[Dict]] = None,
) -> ExecutionResult:
    """
    One-shot entry point: parse + execute in a single call.

    Returns ExecutionResult.answered = True if we got a final answer.
    Returns ExecutionResult.answered = False (and reasons in audit_trail)
    when the question couldn't be answered deterministically.

    `structured_tables` (2026-06-20): preserved table shapes (headers + rows)
    enable exact (metric, year) column lookup — fixes period-collapse bugs.
    """
    plan = parse_question(question)

    # ADVANCED multi-year formula solver (2026-06-21): handles ROA/ROE with
    # averaging, DPO/DSO/DIO, fixed-asset-turnover, 3yr-avg capex%, and
    # YoY-change questions that the single-year DAG executor cannot. Uses the
    # normalized (metric, year) map. Runs FIRST; falls through if not matched.
    if _HAS_ADV and structured_tables:
        try:
            adv_id = _adv.detect_advanced(question)
            if adv_id:
                norm = _tnorm.build_normalized(
                    structured_tables, doc_fiscal_year=fiscal_year
                )
                solved = _adv.solve(question, norm, fiscal_year, raw_text=raw_text)
                if solved is not None:
                    value, unit = solved
                    res = ExecutionResult(answered=True)
                    res.final_value = value
                    res.final_unit = unit
                    res.confidence = 0.9
                    ans = _adv.format_answer(value, unit)
                    citation = f"{company}/{doc_type}/{fiscal_year}/{adv_id}"
                    res.final_answer = f"{ans} [{citation}]"
                    res.audit_trail["advanced_formula"] = adv_id
                    return res
        except Exception:
            logger.debug("[answer_question] advanced solver failed", exc_info=True)

    return execute_plan(
        plan=plan,
        cells=cells or [],
        raw_text=raw_text or "",
        company=company,
        fy=fiscal_year,
        doc_type=doc_type,
        structured_tables=structured_tables or [],
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
