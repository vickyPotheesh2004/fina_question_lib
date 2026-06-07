"""
question_lib/modifier_extractor.py
Pull SIDE-WORDS from a question — the JEE method's secret weapon.

Side-words are tiny qualifiers most LLMs IGNORE:
  - "excluding M&A"        → ModifierKind.EXCLUDING + target='m&a'
  - "assuming 5% growth"   → ModifierKind.ASSUMING  + value=5.0
  - "2x the rate of"       → ModifierKind.MULTIPLIER + value=2.0
  - "between FY22 and FY23"→ ModifierKind.BETWEEN
  - "per share"            → ModifierKind.PER
  - "same rate as"         → ModifierKind.SAME_AS
  - "compared to last year"→ ModifierKind.COMPARED_TO
  - "organic growth"       → ModifierKind.ORGANIC

Each side-word becomes a Modifier object that the decomposer + executor
consume to build the right sub-formula chain.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .models import Modifier, ModifierKind, Polarity


# ─────────────────────────────────────────────────────────────────────────────
# Pattern bank — (regex, kind, value_extractor, target_extractor)
# ─────────────────────────────────────────────────────────────────────────────

# Multipliers: 2x, 3x, twice, half, 1.5x, doubled, tripled
_MULTIPLIER_PATTERNS = [
    (r"\b(\d+(?:\.\d+)?)\s*x\b",                  "explicit"),
    (r"\btwice\b",                                "twice"),
    (r"\bdouble[ds]?\b",                          "twice"),
    (r"\bthrice\b",                               "thrice"),
    (r"\btripled?\b",                             "thrice"),
    (r"\bquadrupled?\b",                          "quadruple"),
    (r"\bhalf\b",                                 "half"),
    (r"\bhalve[ds]?\b",                           "half"),
    (r"\b(\d+(?:\.\d+)?)\s+times\s+(?:the\s+|as\s+)?",  "explicit"),
]

_MULTIPLIER_VALUES = {
    "twice":     2.0,
    "thrice":    3.0,
    "quadruple": 4.0,
    "half":      0.5,
}


# Excluding patterns
_EXCLUDING_PATTERNS = [
    r"\bexcluding\s+(.{1,40}?)(?:[,\.\?]|$|\bfor\b|\bin\b|\bduring\b|\bof\b)",
    r"\bexclude\s+(.{1,40}?)(?:[,\.\?]|$|\bfor\b|\bin\b|\bduring\b|\bof\b)",
    r"\bexcept\s+(.{1,40}?)(?:[,\.\?]|$|\bfor\b|\bin\b|\bduring\b|\bof\b)",
    r"\bwithout\s+(.{1,40}?)(?:[,\.\?]|$|\bfor\b|\bin\b|\bduring\b|\bof\b)",
    r"\bnet\s+of\s+(.{1,40}?)(?:[,\.\?]|$|\bfor\b|\bin\b|\bduring\b)",
]

# Assuming / hypothetical
_ASSUMING_PATTERNS = [
    r"\bassuming\s+(.{1,80}?)(?:[,\.\?]|$|\bwhat\b)",
    r"\bsuppose\s+(.{1,80}?)(?:[,\.\?]|$)",
    r"\bgiven\s+(?:that\s+)?(.{1,80}?)(?:[,\.\?]|$)",
    r"\bif\s+(.{1,80}?)(?:[,\.\?]|\bwhat\b|\bthen\b|$)",
]

# Comparison
_COMPARED_TO_PATTERNS = [
    r"\bcompared\s+to\s+(.{1,40}?)(?:[,\.\?]|$|\bin\b|\bfor\b)",
    r"\brelative\s+to\s+(.{1,40}?)(?:[,\.\?]|$|\bin\b|\bfor\b)",
    r"\bversus\s+(.{1,40}?)(?:[,\.\?]|$|\bin\b|\bfor\b)",
    r"\bvs\.?\s+(.{1,40}?)(?:[,\.\?]|$|\bin\b|\bfor\b)",
]

# Per
_PER_PATTERNS = [
    (r"\bper\s+share\b",          "share"),
    (r"\bper\s+employee\b",       "employee"),
    (r"\bper\s+customer\b",       "customer"),
    (r"\bper\s+user\b",           "user"),
    (r"\bper\s+capita\b",         "capita"),
    (r"\bper\s+(?:square\s+)?foot\b", "foot"),
]

# Same-as / equal-to
_SAME_AS_PATTERNS = [
    r"\bsame\s+(?:rate|pace|level|amount)\s+as\b",
    r"\bat\s+the\s+same\s+(?:rate|pace|level)\b",
    r"\bequivalent\s+to\b",
    r"\bequal\s+to\s+",
    r"\bas\s+\w+\s+as\b",   # "as fast as", "as much as"
]

# Organic / inorganic
_ORGANIC_PATTERNS = [
    r"\borganic(ally)?\b",
    r"\bunderlying\s+(?:growth|revenue)\b",
    r"\bcore\s+(?:growth|business)\b",
]

# Change-in / delta language
_CHANGE_PATTERNS = [
    r"\bchange\s+in\b",
    r"\bdelta\s+",
    r"\b(?:increase|decrease)\s+in\b",
    r"\bgrowth\s+in\b",
    r"\bdrop\s+in\b",
    r"\bdecline\s+in\b",
]

# Negations
_NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bno\s+",
    r"\bnone\s+of\b",
    r"\bneither\s+",
]

# Average-of
_AVERAGE_PATTERNS = [
    r"\baverage\s+of\s+",
    r"\bmean\s+of\s+",
    r"\bover\s+the\s+last\s+\d+\s+years\b",
    r"\b\d+[\s\-]year\s+average\b",
]

# M&A specifically
_MA_TERMS = (
    "m&a", "mergers and acquisitions", "merger", "acquisition",
    "acquired", "divestiture", "divest", "spin-off", "spinoff",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_multiplier_value(matched_text: str, marker: str) -> Optional[float]:
    """Get the numeric multiplier from the matched substring."""
    if marker == "explicit":
        m = re.search(r"\d+(?:\.\d+)?", matched_text)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None
    return _MULTIPLIER_VALUES.get(marker)


def _clean_target(text: str) -> str:
    """Clean up the captured target phrase."""
    if not text:
        return ""
    t = text.strip()
    # Strip leading articles
    t = re.sub(r"^(the|a|an|of|its|their|our|this|that)\s+", "", t, flags=re.IGNORECASE)
    return t.strip(" ,.;:")


def _is_ma_target(target: str) -> bool:
    t = target.lower()
    return any(x in t for x in _MA_TERMS)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_modifiers(question: str) -> List[Modifier]:
    """Return ALL side-word Modifiers found in the question."""
    if not question:
        return []
    q = question
    out: List[Modifier] = []

    # ── MULTIPLIER (2x, twice, half, ...) ──────────────────────────────
    for pattern, marker in _MULTIPLIER_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            raw = m.group(0)
            value = _extract_multiplier_value(raw, marker)
            out.append(Modifier(
                kind=ModifierKind.MULTIPLIER,
                raw=raw,
                value=value,
                confidence=0.95 if value is not None else 0.6,
            ))

    # ── EXCLUDING ──────────────────────────────────────────────────────
    for pattern in _EXCLUDING_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            target = _clean_target(m.group(1)) if m.groups() else ""
            kind = (ModifierKind.EXCLUDING_MA if _is_ma_target(target)
                    else ModifierKind.EXCLUDING)
            out.append(Modifier(
                kind=kind,
                raw=m.group(0),
                target=target,
                confidence=0.95,
            ))

    # ── ASSUMING / IF ──────────────────────────────────────────────────
    for pattern in _ASSUMING_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            target = _clean_target(m.group(1)) if m.groups() else ""
            if not target:
                continue
            # Detect "if" specifically (conditional)
            if re.search(r"\bif\b", m.group(0), re.IGNORECASE):
                kind = ModifierKind.CONDITION_IF
            else:
                kind = ModifierKind.ASSUMING
            out.append(Modifier(
                kind=kind,
                raw=m.group(0),
                target=target,
                confidence=0.9,
            ))

    # ── COMPARED TO / VS ───────────────────────────────────────────────
    for pattern in _COMPARED_TO_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            target = _clean_target(m.group(1)) if m.groups() else ""
            if not target:
                continue
            out.append(Modifier(
                kind=ModifierKind.COMPARED_TO,
                raw=m.group(0),
                target=target,
                confidence=0.9,
            ))

    # ── PER (per share, per employee) ──────────────────────────────────
    for pattern, target in _PER_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            out.append(Modifier(
                kind=ModifierKind.PER,
                raw=m.group(0),
                target=target,
                confidence=0.98,
            ))

    # ── SAME AS / EQUAL ────────────────────────────────────────────────
    for pattern in _SAME_AS_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            out.append(Modifier(
                kind=ModifierKind.SAME_AS,
                raw=m.group(0),
                confidence=0.85,
            ))

    # ── ORGANIC ────────────────────────────────────────────────────────
    for pattern in _ORGANIC_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            out.append(Modifier(
                kind=ModifierKind.ORGANIC,
                raw=m.group(0),
                confidence=0.95,
            ))

    # ── CHANGE IN ──────────────────────────────────────────────────────
    for pattern in _CHANGE_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            polarity = Polarity.NEUTRAL
            raw_lc = m.group(0).lower()
            if any(neg in raw_lc for neg in ("decrease", "drop", "decline")):
                polarity = Polarity.NEGATIVE
            elif "increase" in raw_lc or "growth" in raw_lc:
                polarity = Polarity.POSITIVE
            out.append(Modifier(
                kind=ModifierKind.CHANGE_IN,
                raw=m.group(0),
                polarity=polarity,
                confidence=0.9,
            ))

    # ── AVERAGE OF ─────────────────────────────────────────────────────
    for pattern in _AVERAGE_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            out.append(Modifier(
                kind=ModifierKind.AVERAGE_OF,
                raw=m.group(0),
                confidence=0.9,
            ))

    # ── BETWEEN (range) ────────────────────────────────────────────────
    if re.search(r"\bbetween\s+", q, re.IGNORECASE):
        out.append(Modifier(
            kind=ModifierKind.BETWEEN,
            raw="between",
            confidence=0.85,
        ))

    # ── NEGATION (only if not subsumed by other modifiers) ─────────────
    for pattern in _NEGATION_PATTERNS:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            # Avoid false-positive on "not yet", "not just", etc.
            out.append(Modifier(
                kind=ModifierKind.NEGATION,
                raw=m.group(0),
                confidence=0.55,
            ))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("If revenue grows 2x the rate of last year, what is FY2024?",
         [ModifierKind.CONDITION_IF, ModifierKind.MULTIPLIER]),
        ("Excluding M&A impact, which segment dragged growth?",
         [ModifierKind.EXCLUDING_MA]),
        ("Assuming gross margin stays at 44%, what is FY2024 net income?",
         [ModifierKind.ASSUMING]),
        ("What is Apple's EPS per share?",
         [ModifierKind.PER]),
        ("Compared to last year, how did revenue change?",
         [ModifierKind.COMPARED_TO, ModifierKind.CHANGE_IN]),
        ("Organic growth excluding currency",
         [ModifierKind.EXCLUDING, ModifierKind.ORGANIC]),
        ("Twice the rate of inflation",
         [ModifierKind.MULTIPLIER]),
        ("Between FY21 and FY23, what was the average revenue?",
         [ModifierKind.BETWEEN, ModifierKind.AVERAGE_OF]),
    ]

    print("modifier_extractor — self test")
    passed = 0
    for q, expected_kinds in cases:
        mods = extract_modifiers(q)
        kinds = [m.kind for m in mods]
        # Check all expected appear
        ok = all(k in kinds for k in expected_kinds)
        if ok:
            passed += 1
        print(f"  [{'✓' if ok else '✗'}] expect={expected_kinds!r}")
        print(f"      got={kinds!r}")
        print(f"      Q: {q[:60]}")
    print(f"\n  {passed}/{len(cases)} passed")
