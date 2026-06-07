"""
question_lib/operation_detector.py
Detect the OPERATION within an intent:
    LOOKUP / RATIO / RATIO_PCT / DIFF / SUM / PRODUCT /
    GROWTH_YOY / CAGR / CLASSIFY / EXPLAIN / PROJECT

Operation is the verb. Intent is the goal. They differ:
  - Intent=DECIDE,  Operation=CLASSIFY (apply L02 decision rule)
  - Intent=COMPUTE, Operation=RATIO_PCT (gross margin = gp/rev*100)
  - Intent=PROJECT, Operation=PROJECT  (FY2024 = FY2023 * (1+growth))
"""
from __future__ import annotations

import re
from typing import Optional

from .models import Intent, Operation


# ─────────────────────────────────────────────────────────────────────────────
# Mapping rules (intent + question text) → Operation
# ─────────────────────────────────────────────────────────────────────────────

# Ratio percent: margin, ROE/ROA/ROIC, growth %, anything ending with %
_RATIO_PCT_PATTERNS = (
    r"\bmargin\b",
    r"\b(roa|roe|roic|roce)\b",
    r"\b(growth|change)\s+(?:rate|in|%)\b",
    r"\bgrowth\s+rate\b",
    r"\byoy\b",
    r"\bpercentage\s+(of|change)\b",
    r"\beffective\s+tax\s+rate\b",
    r"\bpayout\s+ratio\b",
    r"\breturn\s+on\b",
)

# Pure ratio (no percent): current ratio, debt/equity, etc.
_RATIO_PATTERNS = (
    r"\b(current|quick|cash|debt[\s\-]to[\s\-]equity|debt[\s\-]to[\s\-]assets|"
    r"asset\s+turnover|inventory\s+turnover|interest\s+coverage)\s+ratio\b",
    r"\bdebt\s*/\s*equity\b",
    r"\bp\s*/\s*e\b",
    r"\bp\s*/\s*b\b",
    r"\bturnover\b",
)

# Growth / CAGR
_GROWTH_PATTERNS = (
    r"\bcagr\b",
    r"\bcompound\s+annual\s+growth\b",
    r"\b\d[\s\-]year\s+growth\b",
)

_YOY_PATTERNS = (
    r"\byoy\b",
    r"\byear[\s\-]over[\s\-]year\b",
    r"\bgrowth\s+from\s+\w+\s+to\s+\w+\b",
)

# Diff
_DIFF_PATTERNS = (
    r"\bchange\s+in\b",
    r"\bdelta\s+",
    r"\bdifference\s+between\b",
    r"\b(?:increased|decreased)\s+by\b",
    r"\bup\s+by\b",
    r"\bdown\s+by\b",
)

# Sum / aggregation
_SUM_PATTERNS = (
    r"\btotal\s+of\b",
    r"\bsum\s+of\b",
    r"\baggregate\s+",
    r"\bcombined\s+",
    r"\bover\s+the\s+last\s+\d+\s+years\b",
)

# Product
_PRODUCT_PATTERNS = (
    r"\bmultipl(?:y|ied)\s+",
    r"\btimes\b",
    r"\bproduct\s+of\b",
)

# Projection
_PROJECT_PATTERNS = (
    r"\bif\s+",
    r"\bassuming\b",
    r"\bproject(ed)?\b",
    r"\bforecast",
    r"\bgoing\s+forward\b",
    r"\bestimate\b",
    r"\bwould\s+be\b",
    r"\bwill\s+be\b",
    r"\bnext\s+(year|quarter)\b",
)


def _matches_any(q: str, patterns) -> bool:
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def detect_operation(question: str, intent: Optional[Intent] = None) -> Operation:
    """Return the most likely Operation for the question.

    `intent` (optional) biases the decision: e.g. intent=DECIDE → CLASSIFY,
    intent=NARRATE → EXPLAIN.
    """
    if not question:
        return Operation.UNKNOWN

    q = question.lower()

    # Strong intent biases first
    if intent == Intent.DECIDE:
        return Operation.CLASSIFY
    if intent == Intent.NARRATE:
        return Operation.EXPLAIN
    if intent == Intent.PROJECT or _matches_any(q, _PROJECT_PATTERNS):
        return Operation.PROJECT

    # Specific operations next
    if _matches_any(q, _GROWTH_PATTERNS):
        return Operation.CAGR
    if _matches_any(q, _YOY_PATTERNS):
        return Operation.GROWTH_YOY
    if _matches_any(q, _RATIO_PCT_PATTERNS):
        return Operation.RATIO_PCT
    if _matches_any(q, _RATIO_PATTERNS):
        return Operation.RATIO
    if _matches_any(q, _DIFF_PATTERNS):
        return Operation.DIFF
    if _matches_any(q, _SUM_PATTERNS):
        return Operation.SUM
    if _matches_any(q, _PRODUCT_PATTERNS):
        return Operation.PRODUCT

    # Default: if intent is EXTRACT/COMPARE/UNKNOWN, treat as LOOKUP
    return Operation.LOOKUP


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("What was Apple's revenue?",                       None,              Operation.LOOKUP),
        ("What is the gross margin?",                       Intent.COMPUTE,    Operation.RATIO_PCT),
        ("Is 3M capital-intensive?",                        Intent.DECIDE,     Operation.CLASSIFY),
        ("What drove margin change?",                       Intent.NARRATE,    Operation.EXPLAIN),
        ("Current ratio?",                                  Intent.COMPUTE,    Operation.RATIO),
        ("Revenue growth YoY?",                              Intent.COMPUTE,    Operation.GROWTH_YOY),
        ("5-year CAGR of revenue",                          Intent.COMPUTE,    Operation.CAGR),
        ("Change in operating income?",                      Intent.COMPUTE,    Operation.DIFF),
        ("If growth stays at 5%, what is FY24 revenue?",    Intent.PROJECT,    Operation.PROJECT),
        ("Sum of dividends over last 3 years",              Intent.COMPUTE,    Operation.SUM),
    ]
    print("operation_detector — self test")
    passed = 0
    for q, intent, expected in cases:
        op = detect_operation(q, intent)
        ok = op == expected
        if ok:
            passed += 1
        print(f"  [{'✓' if ok else '✗'}] expect={expected.value:<10} got={op.value:<10} "
              f"| {q[:50]}")
    print(f"\n  {passed}/{len(cases)} passed")
