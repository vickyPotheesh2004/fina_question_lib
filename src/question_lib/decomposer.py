"""
question_lib/decomposer.py
The JEE banana-peel: decompose a multi-step question into sub-questions.

This is where the side-words + main subject + periods combine into a
SubFormula chain. Each chain is an ordered list of:
    extract → compute → compute → output

Examples:
  "Assuming gross margin grows at 2x revenue growth between FY22 and FY23,
   what would gross margin be in FY24 if revenue grows at the same rate?"

  → SubFormulas:
    1. revenue_growth_yoy   (inputs: rev_2023, rev_2022)
    2. margin_growth_rate   (depends_on: revenue_growth_yoy, multiplier=2)
    3. gross_margin_2023    (inputs: gp_2023, rev_2023)
    4. gross_margin_2024    (depends_on: margin_growth_rate, gross_margin_2023)

100% deterministic. NO LLM.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .formula_matcher    import FormulaMatch, match_formula
from .intent_classifier  import classify_intent
from .modifier_extractor import extract_modifiers
from .operation_detector import detect_operation
from .period_extractor   import extract_periods
from .subject_extractor  import extract_subject, extract_all_subjects
from .models import (
    ExtractRequest, Intent, Modifier, ModifierKind, Operation,
    Period, QuestionPlan, Subject, SubFormula,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def parse_question(question: str) -> QuestionPlan:
    """
    Full parse of a question into a QuestionPlan.

    Pipeline (each step deterministic, single-pass):
      1. classify intent + polarity
      2. extract main subject (metric)
      3. extract periods (time anchors)
      4. extract modifiers (side-words)
      5. detect operation
      6. match formula(s)
      7. build sub-formula chain (decompose if multi-step)
      8. enumerate required raw extracts
    """
    plan = QuestionPlan(raw_question=question or "")

    if not question or not question.strip():
        plan.notes.append("empty question")
        return plan

    # 1. Intent + polarity
    plan.intent, plan.polarity, intent_conf = classify_intent(question)

    # 2. Subject (main metric)
    plan.subject = extract_subject(question)

    # 3. Periods (time anchors)
    plan.periods = extract_periods(question)

    # 4. Modifiers (side-words)
    plan.modifiers = extract_modifiers(question)

    # 5. Operation
    plan.operation = detect_operation(question, plan.intent)

    # 6. Formula match for the main subject
    match: Optional[FormulaMatch] = None
    if plan.subject:
        match = match_formula(plan.subject.metric_id, plan.operation)

    # 7. Build sub-formula chain
    plan.sub_formulas, plan.required_extracts, plan.output_metric, plan.output_unit = (
        _build_sub_formulas(question, plan, match)
    )

    # 8. Confidence + notes
    plan.confidence = _compute_confidence(plan, intent_conf, match)
    plan.audit_trail = {
        "intent_conf":         intent_conf,
        "matched_formula":     match.formula_id if match else None,
        "n_modifiers":         len(plan.modifiers),
        "n_periods":           len(plan.periods),
        "n_sub_formulas":      len(plan.sub_formulas),
        "n_required_extracts": len(plan.required_extracts),
    }

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_sub_formulas(
    question:  str,
    plan:      QuestionPlan,
    main_match: Optional[FormulaMatch],
) -> Tuple[List[SubFormula], List[ExtractRequest], str, str]:
    """
    Build the ordered sub-formula chain + extract requests.

    Strategy:
      - If intent=EXTRACT and subject is a raw metric → 1 extract, no compute
      - If intent=COMPUTE and main_match exists → extract inputs + compute
      - If intent=DECIDE → extract inputs, compute ratio, apply decision rule
      - If intent=PROJECT → 2+ periods needed, chain projection sub-formulas
      - If intent=NARRATE → no formulas, downstream uses narrative_extractor
      - If intent=COMPARE → multiple subjects, each extracted, then compared
    """
    subs:     List[SubFormula]     = []
    extracts: List[ExtractRequest] = []

    primary_period_str = (
        plan.periods[0].fiscal_year if plan.periods and plan.periods[0].fiscal_year
        else ""
    )

    # ── EXTRACT only ───────────────────────────────────────────────────
    if plan.intent == Intent.EXTRACT and plan.subject:
        # Direct value lookup
        extracts.append(ExtractRequest(
            metric_id=plan.subject.metric_id,
            period=primary_period_str,
            label=plan.subject.display_name,
        ))
        output_metric = plan.subject.metric_id
        output_unit = "$M"   # default; refined later by format_lib
        # If subject is a per-share metric, unit is "$"
        if "eps" in plan.subject.metric_id or "per_share" in plan.subject.metric_id:
            output_unit = "$"
        return subs, extracts, output_metric, output_unit

    # ── NARRATE — no formulas, downstream handles narrative ────────────
    if plan.intent == Intent.NARRATE:
        # Still record what subject the narrative is about
        return subs, extracts, plan.subject.metric_id if plan.subject else "", ""

    # ── COMPUTE / DECIDE — match formula + extract inputs ──────────────
    if plan.intent in (Intent.COMPUTE, Intent.DECIDE) and main_match:
        primary_sub = SubFormula(
            name=main_match.formula_id,
            formula_id=main_match.formula_id,
            inputs=list(main_match.inputs),
            operation=main_match.operation,
            period=plan.periods[0] if plan.periods else None,
        )
        subs.append(primary_sub)
        for inp in main_match.inputs:
            extracts.append(ExtractRequest(
                metric_id=inp,
                period=primary_period_str,
                label=f"{inp} for {main_match.formula_id}",
            ))
        # Add decision rule reference if intent=DECIDE
        if plan.intent == Intent.DECIDE:
            plan.decision_rule_id = _suggest_decision_rule(main_match.formula_id)
        return subs, extracts, main_match.formula_id, main_match.output_unit

    # ── PROJECT — needs ≥2 periods + sometimes modifiers ───────────────
    if plan.intent == Intent.PROJECT and plan.subject and main_match:
        # We need: current value + growth rate (from periods or modifier)
        # Step 1: current period value
        if len(plan.periods) >= 1:
            primary_period_str = plan.periods[0].fiscal_year or ""
        # Step 2: prior period (for growth)
        prior_period = ""
        for p in plan.periods[1:]:
            if p.fiscal_year:
                prior_period = p.fiscal_year
                break

        if main_match.operation == Operation.RATIO_PCT:
            # e.g. project gross margin → need gp + rev for current period
            for inp in main_match.inputs:
                extracts.append(ExtractRequest(
                    metric_id=inp,
                    period=primary_period_str,
                    label=f"{inp} (current)",
                ))
                if prior_period:
                    extracts.append(ExtractRequest(
                        metric_id=inp,
                        period=prior_period,
                        label=f"{inp} (prior)",
                    ))

            # Sub-formula: current margin
            subs.append(SubFormula(
                name=f"{main_match.formula_id}_current",
                formula_id=main_match.formula_id,
                inputs=list(main_match.inputs),
                operation=main_match.operation,
                period=plan.periods[0] if plan.periods else None,
            ))
            # Sub-formula: growth rate (from periods or multiplier modifier)
            mult_value = _multiplier_value(plan.modifiers)
            subs.append(SubFormula(
                name="growth_rate",
                formula_id="growth_yoy",
                inputs=["t", "t_minus_1"],
                operation=Operation.GROWTH_YOY,
                modifiers=[m for m in plan.modifiers if m.kind == ModifierKind.MULTIPLIER],
                notes=f"multiplier={mult_value}" if mult_value else "",
            ))
            # Sub-formula: projected
            subs.append(SubFormula(
                name=f"{main_match.formula_id}_projected",
                formula_id="projection_compound",
                depends_on=[f"{main_match.formula_id}_current", "growth_rate"],
                operation=Operation.PROJECT,
            ))
        else:
            # Simpler: project a raw value
            extracts.append(ExtractRequest(
                metric_id=plan.subject.metric_id,
                period=primary_period_str,
                label=f"{plan.subject.metric_id} (current)",
            ))
            if prior_period:
                extracts.append(ExtractRequest(
                    metric_id=plan.subject.metric_id,
                    period=prior_period,
                    label=f"{plan.subject.metric_id} (prior)",
                ))
            subs.append(SubFormula(
                name="growth_rate",
                formula_id="growth_yoy",
                operation=Operation.GROWTH_YOY,
            ))
            subs.append(SubFormula(
                name=f"{plan.subject.metric_id}_projected",
                formula_id="projection_compound",
                depends_on=["growth_rate"],
                operation=Operation.PROJECT,
            ))

        return subs, extracts, f"{main_match.formula_id if main_match else plan.subject.metric_id}_projected", main_match.output_unit

    # ── COMPARE — pull all subjects; comparison done downstream ────────
    if plan.intent == Intent.COMPARE:
        subjects = extract_all_subjects(question, max_subjects=4)
        if not subjects and plan.subject:
            subjects = [plan.subject]
        for s in subjects:
            extracts.append(ExtractRequest(
                metric_id=s.metric_id,
                period=primary_period_str,
                label=s.display_name,
            ))
        return subs, extracts, "comparison", ""

    # ── Fallback ───────────────────────────────────────────────────────
    if plan.subject:
        extracts.append(ExtractRequest(
            metric_id=plan.subject.metric_id,
            period=primary_period_str,
            label=plan.subject.display_name,
        ))
        return subs, extracts, plan.subject.metric_id, ""

    return subs, extracts, "", ""


def _multiplier_value(modifiers: List[Modifier]) -> Optional[float]:
    for m in modifiers:
        if m.kind == ModifierKind.MULTIPLIER and m.value is not None:
            return m.value
    return None


def _suggest_decision_rule(formula_id: str) -> str:
    """Map common formulas to logic_lib L02 decision rule ids."""
    return {
        "capex_to_revenue":   "capital_intensity_class",
        "capital_intensity":  "capital_intensity_class",
        "current_ratio":      "liquidity_current_ratio",
        "quick_ratio":        "liquidity_quick_ratio",
        "cash_ratio":         "liquidity_cash_ratio",
        "debt_to_equity":     "leverage_debt_equity",
        "interest_coverage":  "solvency_interest_coverage",
        "net_margin":         "profitability_net_margin",
        "roe":                "profitability_roe",
        "roa":                "profitability_net_margin",   # close approx
        "free_cash_flow":     "fcf_positive",
        "payout_ratio":       "dividend_sustainability",
        "working_capital":    "working_capital_health",
    }.get(formula_id, "yes_no_threshold")


def _compute_confidence(
    plan: QuestionPlan,
    intent_conf: float,
    match: Optional[FormulaMatch],
) -> float:
    """Combined plan confidence (0..1)."""
    base = intent_conf
    if plan.subject:
        base = (base + plan.subject.confidence) / 2
    if plan.periods:
        base += 0.05
    if match:
        base += 0.05
    if plan.intent == Intent.UNKNOWN:
        base *= 0.5
    return min(1.0, max(0.0, base))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        "What was Apple's revenue in FY2023?",
        "What was 3M's gross margin in FY2022?",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "What drove operating margin change in FY2022?",
        "If revenue grows at 5% per year, what would FY2024 revenue be?",
        "Compare Apple's revenue vs Microsoft's revenue in FY2023",
        "Which segment dragged down 3M's growth in 2022?",
        "What is Apple's current ratio?",
    ]
    print("decomposer — self test\n")
    for q in cases:
        plan = parse_question(q)
        print(f"Q: {q}")
        print(f"   intent     = {plan.intent.value}")
        print(f"   subject    = {plan.subject.metric_id if plan.subject else None}")
        print(f"   operation  = {plan.operation.value}")
        print(f"   periods    = {[p.fiscal_year for p in plan.periods]}")
        print(f"   modifiers  = {[m.kind for m in plan.modifiers]}")
        print(f"   sub_forms  = {[s.name for s in plan.sub_formulas]}")
        print(f"   extracts   = {[(e.metric_id, e.period) for e in plan.required_extracts]}")
        print(f"   conf       = {plan.confidence:.2f}")
        if plan.decision_rule_id:
            print(f"   rule       = {plan.decision_rule_id}")
        print()
