"""
question_lib/intent_classifier.py
Classify a question into one of 6 top-level intents:
    EXTRACT  / COMPUTE  / DECIDE  / COMPARE  / NARRATE  / PROJECT

Pattern-based, deterministic, sub-millisecond. NO LLM.

Returns: (Intent, Polarity, confidence)
"""
from __future__ import annotations

import re
from typing import Tuple

from .models import Intent, Polarity


# ─────────────────────────────────────────────────────────────────────────────
# Pattern bank — ordered (most specific first, generic last)
# ─────────────────────────────────────────────────────────────────────────────

# PROJECT — hypothetical / "if/assuming/would be"
_PROJECT_PATTERNS = (
    r"\bif\s+.*\b(grow|grew|grows|growth|increase|decrease|reach|hit|stay|continue)",
    r"\bassuming\s+",
    r"\bwhat\s+would\s+",
    r"\bwhat\s+will\s+",
    r"\b(project|projected|forecast|estimate|extrapolate)d?\s+",
    r"\bgoing\s+forward\b",
    r"\bin\s+(?:fy\s*)?20[2-3]\d\s+if\b",
    r"\bnext\s+(year|quarter|fiscal\s+year)\b",
)

# DECIDE — yes/no, classification, "is X Y?"
_DECIDE_PATTERNS = (
    r"\b(is|are|was|were|does|do|did|has|have|had|can|could|should|would)\s+\w+(\s+\w+){0,3}\s+"
    r"(?:a|an|the\s+)?\s*"
    r"(?:capital[\s\-]intensive|liquid|leveraged|profitable|solvent|healthy|sustainable|"
    r"strong|weak|risky|safe|undervalued|overvalued|attractive|distressed|going\s+concern)",
    r"\bcapital[\s\-]intensive\b",
    r"\b(is|does)\s+\w+(\s+\w+)?\s+meet\s+(its\s+)?(covenant|guidance|target)",
    r"\b(can|could)\s+\w+\s+(?:cover|pay|service)\s+(its\s+)?debt",
    r"\b(yes\s+or\s+no|true\s+or\s+false)\b",
    r"\bis\s+this\s+a\s+",
)

# NARRATE — "what drove X", "why did Y", "explain Z", "what caused"
_NARRATE_PATTERNS = (
    r"\bwhat\s+drove\b",
    r"\bwhat\s+caused\b",
    r"\bwhat\s+led\s+to\b",
    r"\bwhy\s+did\b",
    r"\bwhy\s+is\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bdiscuss\b",
    r"\breason(s)?\s+for\b",
    r"\bdriver(s)?\s+of\b",
    r"\bfactor(s)?\s+(that|behind|contributing)\b",
    r"\bwhat\s+factor(s)?\b",
    r"\bsummari[sz]e\b",
    r"\bnarrative\s+of\b",
    r"\bwhich\s+(segment|business|division)\s+(dragged|had|with|showed|drove)",
    r"\bwhich\s+(segment|business|division)",
)

# COMPARE — across companies / periods / peers
_COMPARE_PATTERNS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\b\s+vs\.?\s+",
    r"\bbetween\s+\w+\s+and\s+\w+\b",
    r"\b(higher|lower|larger|smaller|better|worse|more|less)\s+than\b",
    r"\brelative\s+to\b",
    r"\b(top|leading|biggest|largest)\s+\w+",
    r"\bpeer(s)?\b",
    r"\bcompetitor(s)?\b",
    r"\boutperform(ed)?\b",
    r"\bunderperform(ed)?\b",
)

# COMPUTE — ratios, margins, growth, derived metrics
_COMPUTE_PATTERNS = (
    r"\b(gross|operating|net|ebitda|ebit)\s+margin\b",
    r"\b(roa|roe|roic|roce)\b",
    r"\breturn\s+on\s+(assets|equity|capital|investment)\b",
    r"\b(current|quick|cash|debt[\s\-]to[\s\-]equity|debt[\s\-]to[\s\-]assets)\s+ratio\b",
    r"\binterest\s+coverage\b",
    r"\b(growth|change|increase|decrease)\s+(rate|in)\b",
    r"\byoy\b",
    r"\byear[\s\-]over[\s\-]year\b",
    r"\bcagr\b",
    r"\b(working\s+capital|free\s+cash\s+flow|fcf)\s+(margin|yield|ratio)\b",
    r"\bturnover\s+(ratio|rate|days)\b",
    r"\bdso\b",
    r"\bdio\b",
    r"\bdpo\b",
    r"\bdays\s+(sales|inventory|payable)\s+(outstanding|on\s+hand)\b",
    r"\bccc\b|\bcash\s+conversion\s+cycle\b",
    r"\bp/e|\bprice\s+to\s+earnings\b",
    r"\beps\b",
    r"\bearnings\s+per\s+share\b",
    r"\b(beta|sharpe|sortino|treynor)\b",
    r"\b(var|value\s+at\s+risk)\b",
    r"\bdcf\b|\bdiscounted\s+cash\s+flow\b",
    r"\bwacc\b",
    r"\bnpv\b|\bnet\s+present\s+value\b",
    r"\birr\b|\binternal\s+rate\s+of\s+return\b",
)

# EXTRACT — "what was X", "how much", direct value lookup
_EXTRACT_PATTERNS = (
    r"\bwhat\s+was\b",
    r"\bwhat\s+is\b",
    r"\bhow\s+much\b",
    r"\bhow\s+many\b",
    r"\bamount\s+of\b",
    r"\btotal\s+\w+",
    r"\bvalue\s+of\b",
    r"\breport(ed)?\s+",
    r"\bin\s+fy\s*20\d\d\b",
    r"\bat\s+(?:the\s+)?(?:year[\s\-]end|fiscal\s+year[\s\-]end|end\s+of)\b",
)


# ─────────────────────────────────────────────────────────────────────────────
# Polarity — direction of decision questions
# ─────────────────────────────────────────────────────────────────────────────

_POLARITY_POS = (
    r"\b(healthy|good|strong|excellent|robust|sustainable|profitable|"
    r"liquid|solvent|safe|stable|growing|improving|attractive|"
    r"undervalued|outperform)\b"
)
_POLARITY_NEG = (
    r"\b(weak|bad|poor|risky|distressed|unsustainable|declining|"
    r"shrinking|deteriorating|overvalued|underperform|unhealthy|"
    r"unprofitable|illiquid|insolvent)\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _any_match(patterns: Tuple[str, ...], q: str) -> bool:
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


def classify_intent(question: str) -> Tuple[Intent, Polarity, float]:
    """Return (Intent, Polarity, confidence).

    Order matters: PROJECT and NARRATE checked before COMPARE / COMPUTE
    because they're more specific (override).
    """
    if not question or not question.strip():
        return Intent.UNKNOWN, Polarity.NEUTRAL, 0.0

    q = question.lower().strip()

    # 1. PROJECT (hypothetical)
    if _any_match(_PROJECT_PATTERNS, q):
        return Intent.PROJECT, _detect_polarity(q), 0.92

    # 2. NARRATE (causal / qualitative)
    if _any_match(_NARRATE_PATTERNS, q):
        return Intent.NARRATE, _detect_polarity(q), 0.90

    # 3. DECIDE (yes/no + classification)
    if _any_match(_DECIDE_PATTERNS, q):
        return Intent.DECIDE, _detect_polarity(q), 0.88

    # 4. COMPARE (cross-entity / cross-period)
    if _any_match(_COMPARE_PATTERNS, q):
        return Intent.COMPARE, _detect_polarity(q), 0.86

    # 5. COMPUTE (derived metrics, ratios)
    if _any_match(_COMPUTE_PATTERNS, q):
        return Intent.COMPUTE, _detect_polarity(q), 0.85

    # 6. EXTRACT (direct lookup) — last because most generic
    if _any_match(_EXTRACT_PATTERNS, q):
        return Intent.EXTRACT, Polarity.NEUTRAL, 0.80

    # 7. Fallback: try one more pass for "what" + metric → likely EXTRACT
    if "what" in q and any(
        kw in q for kw in (
            "revenue", "income", "earnings", "capex", "cash flow",
            "assets", "liabilities", "equity", "debt", "margin",
        )
    ):
        return Intent.EXTRACT, Polarity.NEUTRAL, 0.65

    return Intent.UNKNOWN, Polarity.NEUTRAL, 0.30


def _detect_polarity(q: str) -> Polarity:
    """Detect positive/negative framing within a question."""
    if re.search(_POLARITY_NEG, q):
        return Polarity.NEGATIVE
    if re.search(_POLARITY_POS, q):
        return Polarity.POSITIVE
    return Polarity.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("What was Apple's revenue in FY2023?",                        Intent.EXTRACT),
        ("What was 3M's gross margin in FY2022?",                       Intent.COMPUTE),
        ("Is 3M a capital-intensive business in FY2022?",               Intent.DECIDE),
        ("What drove operating margin change in FY2022?",               Intent.NARRATE),
        ("Compare Apple vs Microsoft revenue in FY2023",                Intent.COMPARE),
        ("If revenue grows at 5%, what would FY2024 revenue be?",       Intent.PROJECT),
        ("Which segment dragged down 3M's growth in 2022?",             Intent.NARRATE),
        ("How much capex did 3M spend in FY2018?",                       Intent.EXTRACT),
        ("What is Apple's current ratio?",                              Intent.COMPUTE),
        ("Is Apple's liquidity healthy?",                               Intent.DECIDE),
    ]

    print("intent_classifier — self test")
    passed = 0
    for q, expected in cases:
        intent, pol, conf = classify_intent(q)
        ok = "✓" if intent == expected else "✗"
        if intent == expected:
            passed += 1
        print(f"  [{ok}] expect={expected.value:<8} got={intent.value:<8} "
              f"pol={pol.value:<8} conf={conf:.2f}  | {q[:60]}")
    print(f"\n  {passed}/{len(cases)} passed")
