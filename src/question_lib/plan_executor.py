"""
question_lib/plan_executor.py
Execute a QuestionPlan deterministically using the 7 support libraries.

Pipeline:
  1. Resolve each ExtractRequest    → extract_lib.resolve_metric (or raw_text)
  2. Execute each SubFormula in DAG order
  3. If decision_rule_id set        → logic_lib.fire(rule, **inputs)
  4. Apply verify_lib sanity bounds  → block on abstain
  5. Format via format_lib           → final answer string

100% deterministic. NO LLM. Never raises.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .dependency_graph import execution_order
from .models import (
    ExecutionResult, ExtractRequest, Operation,
    QuestionPlan, SubFormula,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional lib imports — all soft
# ─────────────────────────────────────────────────────────────────────────────

try:
    from extract_lib.synonyms import METRIC_SYNONYMS as _EXT_SYNS
    from extract_lib.resolver  import resolve_metric as _resolve_metric
    _HAS_EXTRACT = True
except Exception:
    _EXT_SYNS = {}
    _resolve_metric = None
    _HAS_EXTRACT = False

try:
    import maths_lib as _maths_lib
    _HAS_MATHS = True
except Exception:
    _maths_lib = None
    _HAS_MATHS = False

try:
    from logic_lib import fire as _logic_fire
    _HAS_LOGIC = True
except Exception:
    _logic_fire = None
    _HAS_LOGIC = False

try:
    from format_lib import render as _format_render
    _HAS_FORMAT = True
except Exception:
    _format_render = None
    _HAS_FORMAT = False

try:
    from verify_lib import verify_answer as _verify
    _HAS_VERIFY = True
except Exception:
    _verify = None
    _HAS_VERIFY = False


# ─────────────────────────────────────────────────────────────────────────────
# Number scan (reused style from pipeline)
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"\(?\$?\s*\-?\s*\d+(?:,\d{3})*(?:\.\d+)?\)?"
    r"(?:\s*(?:million|billion|thousand|bn|mn|m|b))?",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[,;:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_number(token: str) -> Optional[float]:
    if not token:
        return None
    raw = str(token).strip()
    neg = "(" in raw and ")" in raw
    raw = re.sub(r"[\$\s\(\)%,]", "", raw)
    m = re.match(r"^([-\d\.]+)\s*(million|billion|thousand|bn|mn|m|b)?$",
                 raw, re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if neg:
        v = -abs(v)
    suf = (m.group(2) or "").lower()
    if suf in ("billion", "bn", "b"):
        v *= 1000.0
    elif suf == "thousand":
        v /= 1000.0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Raw extraction (cells first, raw_text fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_value(
    metric_id: str,
    period:    str,
    cells:     List[Dict],
    raw_text:  str,
) -> Optional[float]:
    """Try table_cells via extract_lib; fall back to raw_text scan."""
    # Path 1: extract_lib on cells
    if _HAS_EXTRACT and cells:
        try:
            r = _resolve_metric(metric_id, cells, period or "")
            if r and r.valid and r.value is not None:
                return float(r.value)
        except Exception:
            logger.debug("[plan_executor] extract_lib failed", exc_info=True)

    # Path 2: raw_text scan via synonyms
    if not raw_text:
        return None
    syn = _EXT_SYNS.get(metric_id, {}) if _HAS_EXTRACT else {}
    positives = syn.get("positive", [])
    if not positives:
        return None

    raw_norm = _norm(raw_text)
    for synonym in sorted(positives, key=len, reverse=True):
        s = _norm(synonym)
        idx = raw_norm.find(s)
        if idx < 0:
            continue
        window = raw_norm[idx + len(s): idx + len(s) + 300]
        m = _NUMBER_RE.search(window)
        if m:
            v = _parse_number(m.group(0))
            if v is not None and abs(v) > 0.5:    # avoid pure-zero noise
                return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sub-formula execution
# ─────────────────────────────────────────────────────────────────────────────

def _execute_sub(
    sub:           SubFormula,
    extracted:     Dict[str, float],
    intermediate:  Dict[str, float],
    multiplier:    float = 1.0,
) -> Optional[float]:
    """Compute one SubFormula. Returns the value or None on failure."""
    fid = sub.formula_id

    # Resolve inputs (from extracted OR intermediate)
    inputs = []
    for name in sub.inputs:
        if name in extracted:
            inputs.append(extracted[name])
        elif name in intermediate:
            inputs.append(intermediate[name])
        else:
            # Look for a partial match (e.g. "rev_2023" → "revenue")
            base = name.split("_")[0]
            matched = next(
                (v for k, v in extracted.items() if k.startswith(base)),
                None,
            )
            if matched is not None:
                inputs.append(matched)
            else:
                logger.debug("[plan_executor] missing input %s for %s", name, fid)
                return None

    # Dependencies
    for dep in sub.depends_on:
        if dep not in intermediate:
            logger.debug("[plan_executor] unresolved dependency %s for %s", dep, fid)
            return None

    # Operation execution
    op = sub.operation
    try:
        if op == Operation.RATIO_PCT and len(inputs) >= 2:
            return abs(inputs[0]) / abs(inputs[1]) * 100.0
        if op == Operation.RATIO and len(inputs) >= 2:
            return abs(inputs[0]) / abs(inputs[1])
        if op == Operation.DIFF and len(inputs) >= 2:
            return inputs[0] - inputs[1]
        if op == Operation.SUM:
            return sum(inputs)
        if op == Operation.PRODUCT and inputs:
            r = 1.0
            for x in inputs:
                r *= x
            return r
        if op == Operation.GROWTH_YOY and len(inputs) >= 2:
            return (inputs[0] - inputs[1]) / abs(inputs[1]) * 100.0
        if op == Operation.CAGR and len(inputs) >= 2:
            n = max(1, (sub.notes.count("year") or 1))
            return (inputs[0] / abs(inputs[1])) ** (1 / n) - 1
        if op == Operation.PROJECT:
            # Use intermediate growth_rate and current value
            current = intermediate.get(
                next((d for d in sub.depends_on if "current" in d), ""),
                None,
            )
            growth = intermediate.get("growth_rate", None)
            if current is not None and growth is not None:
                return current * (1.0 + growth / 100.0 * multiplier)
        if op == Operation.LOOKUP and inputs:
            return inputs[0]
    except (ZeroDivisionError, TypeError, ValueError):
        logger.exception("[plan_executor] sub-formula compute failed: %s", fid)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def execute_plan(
    plan:     QuestionPlan,
    cells:    List[Dict],
    raw_text: str,
    company:  str = "",
    fy:       str = "",
    doc_type: str = "",
) -> ExecutionResult:
    """
    Execute the plan deterministically using all available libs.

    Never raises.
    """
    result = ExecutionResult(answered=False)
    result.used_libs = [
        x for x, ok in (
            ("extract_lib", _HAS_EXTRACT),
            ("maths_lib",   _HAS_MATHS),
            ("logic_lib",   _HAS_LOGIC),
            ("format_lib",  _HAS_FORMAT),
            ("verify_lib",  _HAS_VERIFY),
        ) if ok
    ]

    if not plan or not plan.required_extracts and not plan.sub_formulas:
        result.audit_trail["reason"] = "empty_plan"
        return result

    # 1. Resolve all ExtractRequests
    extracted: Dict[str, float] = {}
    for req in plan.required_extracts:
        val = _extract_value(req.metric_id, req.period, cells, raw_text)
        if val is not None:
            key = f"{req.metric_id}_{req.period}" if req.period else req.metric_id
            extracted[key] = val
            # Also store under bare metric id (for first hit)
            extracted.setdefault(req.metric_id, val)
    result.audit_trail["extracted"] = dict(extracted)

    # 2. Execute SubFormulas in topological order
    multiplier = 1.0
    from .models import ModifierKind
    for m in plan.modifiers:
        if m.kind == ModifierKind.MULTIPLIER and m.value is not None:
            multiplier = float(m.value)
            break

    sorted_subs, cycles = execution_order(plan.sub_formulas)
    if cycles:
        result.audit_trail["cycle_in_dag"] = cycles
        return result

    intermediate: Dict[str, float] = {}
    for sub in sorted_subs:
        val = _execute_sub(sub, extracted, intermediate, multiplier=multiplier)
        if val is None:
            result.audit_trail.setdefault("failed_subs", []).append(sub.name)
            continue
        intermediate[sub.name] = val
    result.intermediate_values = dict(intermediate)

    # 3. Final value resolution
    final_value: Optional[float] = None
    if plan.sub_formulas:
        # Last sub-formula computed = final answer
        last = sorted_subs[-1] if sorted_subs else None
        if last and last.name in intermediate:
            final_value = intermediate[last.name]
    elif extracted:
        # Direct extraction
        if plan.subject and plan.subject.metric_id in extracted:
            final_value = extracted[plan.subject.metric_id]

    if final_value is None:
        result.audit_trail["reason"] = "no_final_value"
        return result

    # 4. Decision-rule firing (logic_lib L02)
    classification = ""
    if plan.decision_rule_id and _HAS_LOGIC and _logic_fire is not None:
        try:
            # Heuristic: the rule input name is the primary metric being judged
            # capex_to_revenue, current_ratio, etc.
            rule_input_name = _guess_rule_input_name(plan.decision_rule_id)
            r = _logic_fire(plan.decision_rule_id, **{rule_input_name: final_value})
            if r and getattr(r, "fired", False):
                classification = str(getattr(r, "output", ""))
                result.audit_trail["decision_branch"] = getattr(r, "branch", "")
        except Exception:
            logger.exception("[plan_executor] logic_lib.fire failed")

    # 5. verify_lib sanity check (block if abstain)
    if _HAS_VERIFY and _verify is not None:
        try:
            v = _verify(plan.output_metric or "value",
                        final_value,
                        confidence=plan.confidence)
            if getattr(v, "abstain", False) or not getattr(v, "ok", True):
                result.audit_trail["verify_abstain"] = list(getattr(v, "reasons", []))
                result.audit_trail["reason"] = "verify_lib_abstain"
                return result
        except Exception:
            logger.debug("[plan_executor] verify_lib failed", exc_info=True)

    # 6. Format final answer
    unit = plan.output_unit or ""
    if classification:
        # Decision-style output
        cap_class = classification[:1].upper() + classification[1:]
        unit_suffix = "%" if "%" in unit or "_pct" in (plan.operation.value or "") else ""
        result.final_answer = (
            f"{cap_class}. {plan.subject.display_name if plan.subject else plan.output_metric}: "
            f"{final_value:.2f}{unit_suffix}."
        )
    else:
        # Numeric output
        formatted = _format_safe(final_value, unit)
        result.final_answer = formatted

    citation = f"{company}/{doc_type}/{fy}/{plan.output_metric}"
    if result.final_answer:
        result.final_answer = f"{result.final_answer} [{citation}]"

    result.final_value = final_value
    result.final_unit = unit
    result.answered = True
    result.confidence = plan.confidence
    return result


def _format_safe(value: float, unit: str) -> str:
    if _HAS_FORMAT and _format_render is not None:
        try:
            r = _format_render("default_number", value)
            if r is not None:
                txt = getattr(r, "text", None) or getattr(r, "value", None) or str(r)
                # Append unit if not already encoded
                if unit == "%" and "%" not in str(txt):
                    return f"{txt}%"
                return str(txt)
        except Exception:
            pass
    # Fallback formatting
    if unit == "%":
        return f"{value:.2f}%"
    if unit in ("x", ""):
        return f"{value:.2f}"
    if unit in ("$", "$M"):
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.2f}M" if unit == "$M" else f"{sign}${abs(value):.2f}"
    return f"{value:.2f} {unit}"


def _guess_rule_input_name(rule_id: str) -> str:
    """Map decision rule id → its kwarg name."""
    return {
        "capital_intensity_class":     "capex_to_revenue",
        "liquidity_current_ratio":     "current_ratio",
        "liquidity_quick_ratio":       "quick_ratio",
        "liquidity_cash_ratio":        "cash_ratio",
        "leverage_debt_equity":        "debt_to_equity",
        "leverage_debt_ebitda":        "debt_to_ebitda",
        "solvency_interest_coverage":  "interest_coverage",
        "profitability_net_margin":    "net_margin",
        "profitability_roe":           "roe",
        "fcf_positive":                "fcf",
        "dividend_sustainability":     "payout_ratio",
        "working_capital_health":      "working_capital",
    }.get(rule_id, "value")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .decomposer import parse_question

    print("plan_executor — self test (offline, libs may be missing)")
    raw_text = (
        "Capital expenditures for FY2022 totaled $1,749 million. "
        "Net sales for the year were $34,229 million. "
        "Net income was $5,791 million. "
    )

    cases = [
        "What was 3M's revenue in FY2022?",
        "What was 3M's net income in FY2022?",
        "Is 3M a capital-intensive business in FY2022?",
    ]
    for q in cases:
        plan = parse_question(q)
        res = execute_plan(plan, cells=[], raw_text=raw_text,
                            company="3M", fy="FY2022", doc_type="10-K")
        print(f"\nQ: {q}")
        print(f"   intent      = {plan.intent.value}")
        print(f"   subject     = {plan.subject.metric_id if plan.subject else None}")
        print(f"   extracted   = {res.audit_trail.get('extracted', {})}")
        print(f"   answered    = {res.answered}")
        print(f"   final       = {res.final_answer}")
        print(f"   confidence  = {res.confidence:.2f}")
