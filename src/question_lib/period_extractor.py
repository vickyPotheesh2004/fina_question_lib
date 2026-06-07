"""
question_lib/period_extractor.py
Extract time anchors from a question.

Handles:
  - "FY2022", "FY 2022", "fiscal year 2022", "2022"
  - "Q1 FY2023", "Q4 2023", "fourth quarter 2023"
  - "year-end FY2022", "end of FY2022"
  - "between FY2021 and FY2022"
  - "trailing twelve months", "TTM", "LTM"
  - "last year", "prior year", "next year", "this year"
  - calendar context: "in 2023", "during 2022"
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional

from .models import Period


# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

# FY-prefixed years
_RE_FY        = re.compile(r"\bfy\s*([12]\d{3})\b", re.IGNORECASE)
_RE_FY_DOT    = re.compile(r"\bfy['\s]?(\d{2})\b", re.IGNORECASE)   # FY22, FY'22
_RE_FISCAL    = re.compile(r"\bfiscal\s+year\s+([12]\d{3})\b", re.IGNORECASE)

# Quarter patterns
_RE_QUARTER_FY = re.compile(
    r"\b(q[1-4]|1q|2q|3q|4q|first\s+quarter|second\s+quarter|"
    r"third\s+quarter|fourth\s+quarter)\s+"
    r"(?:fy\s*)?([12]\d{3})\b",
    re.IGNORECASE,
)
_RE_QUARTER = re.compile(
    r"\b(q[1-4]|1q|2q|3q|4q)\s+([12]\d{3})\b",
    re.IGNORECASE,
)

# Plain calendar year (only treat as period if context indicates)
_RE_CAL_YEAR = re.compile(
    r"\b(?:in|during|for|of)\s+([12]\d{3})\b",
    re.IGNORECASE,
)
_RE_BARE_YEAR = re.compile(r"\b([12]\d{3})\b")

# Range patterns — "between FY22 and FY23"
_RE_BETWEEN = re.compile(
    r"\bbetween\s+"
    r"(?:fy\s*)?([12]\d{3}|['\s]?\d{2})"
    r"\s+(?:and|to|-)\s+"
    r"(?:fy\s*)?([12]\d{3}|['\s]?\d{2})",
    re.IGNORECASE,
)

# TTM / LTM
_RE_TTM = re.compile(
    r"\b(ttm|trailing\s+twelve\s+months|trailing\s+12\s+months|"
    r"ltm|last\s+twelve\s+months|last\s+12\s+months)\b",
    re.IGNORECASE,
)

# Year-end markers
_RE_YEAR_END = re.compile(
    r"\b(?:year[\s\-]end|fiscal\s+year[\s\-]end|end\s+of\s+(?:the\s+)?fiscal\s+year|"
    r"end\s+of\s+fy|year\s+ended)\b",
    re.IGNORECASE,
)

# Relative time
_RE_LAST_YEAR  = re.compile(r"\b(?:last|prior|previous)\s+year\b", re.IGNORECASE)
_RE_THIS_YEAR  = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
_RE_NEXT_YEAR  = re.compile(r"\bnext\s+(?:year|fiscal\s+year)\b", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_QUARTER_MAP = {
    "q1": "Q1", "1q": "Q1", "first quarter":  "Q1",
    "q2": "Q2", "2q": "Q2", "second quarter": "Q2",
    "q3": "Q3", "3q": "Q3", "third quarter":  "Q3",
    "q4": "Q4", "4q": "Q4", "fourth quarter": "Q4",
}


def _normalise_year(token: str) -> Optional[str]:
    """'22' → '2022', '99' → '1999', '2023' → '2023'."""
    t = re.sub(r"[^\d]", "", token or "")
    if not t.isdigit():
        return None
    if len(t) == 4:
        return t
    if len(t) == 2:
        n = int(t)
        # Pivot at 50 — '49 → 2049, '50 → 1950
        return f"20{n:02d}" if n < 50 else f"19{n:02d}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_periods(question: str, reference_year: Optional[int] = None) -> List[Period]:
    """
    Extract ALL Period anchors from `question`.

    `reference_year` is used to resolve "last year" / "this year" relative
    references. Defaults to current calendar year.

    Returns periods in order of appearance.
    """
    if not question:
        return []

    ref_year = reference_year or date.today().year
    q = question

    out: List[Period] = []

    # 1. RANGE "between FY22 and FY23" — process FIRST so we don't
    #    pick up the constituent years twice.
    m_between = _RE_BETWEEN.search(q)
    if m_between:
        y1 = _normalise_year(m_between.group(1))
        y2 = _normalise_year(m_between.group(2))
        if y1 and y2:
            p1 = Period(raw=m_between.group(0), fiscal_year=y1, is_range=True)
            p2 = Period(raw=m_between.group(0), fiscal_year=y2, is_range=True)
            p1.range_end = p2
            out.append(p1)
            out.append(p2)

    # 2. Quarter + year ("Q4 2022", "Q1 FY2023")
    for m in _RE_QUARTER_FY.finditer(q):
        q_tok = m.group(1).lower()
        y_tok = m.group(2)
        out.append(Period(
            raw=m.group(0),
            fiscal_year=_normalise_year(y_tok),
            quarter=_QUARTER_MAP.get(q_tok, q_tok.upper()),
        ))
    for m in _RE_QUARTER.finditer(q):
        q_tok = m.group(1).lower()
        y_tok = m.group(2)
        # avoid double-count if already matched by _RE_QUARTER_FY
        if any(_span_overlap(m.span(), p.raw, q) for p in out):
            continue
        out.append(Period(
            raw=m.group(0),
            fiscal_year=_normalise_year(y_tok),
            quarter=_QUARTER_MAP.get(q_tok, q_tok.upper()),
        ))

    # 3. FY-prefixed years
    for m in _RE_FY.finditer(q):
        y = _normalise_year(m.group(1))
        if y and not _already_have_year(out, y):
            out.append(Period(raw=m.group(0), fiscal_year=y))
    for m in _RE_FY_DOT.finditer(q):
        y = _normalise_year(m.group(1))
        if y and not _already_have_year(out, y):
            out.append(Period(raw=m.group(0), fiscal_year=y))
    for m in _RE_FISCAL.finditer(q):
        y = _normalise_year(m.group(1))
        if y and not _already_have_year(out, y):
            out.append(Period(raw=m.group(0), fiscal_year=y))

    # 4. Calendar year (in 2023, during 2022)
    for m in _RE_CAL_YEAR.finditer(q):
        y = _normalise_year(m.group(1))
        if y and not _already_have_year(out, y):
            out.append(Period(
                raw=m.group(0),
                fiscal_year=y,
                calendar_year=int(y),
            ))

    # 5. Bare year (only if NOTHING else matched, to avoid noise)
    if not out:
        for m in _RE_BARE_YEAR.finditer(q):
            y = _normalise_year(m.group(1))
            yi = int(y) if y else 0
            # Only accept "real" year range to avoid 1500/3000 false positives
            if y and 1990 <= yi <= 2050:
                out.append(Period(raw=m.group(0), fiscal_year=y, calendar_year=yi))

    # 6. Year-end flag
    if _RE_YEAR_END.search(q) and out:
        out[0].is_year_end = True

    # 7. TTM / LTM
    if _RE_TTM.search(q):
        out.append(Period(raw="TTM", is_ttm=True))

    # 8. Relative ("last year", "this year")
    if _RE_LAST_YEAR.search(q):
        out.append(Period(raw="last year", fiscal_year=str(ref_year - 1)))
    if _RE_THIS_YEAR.search(q):
        out.append(Period(raw="this year", fiscal_year=str(ref_year)))
    if _RE_NEXT_YEAR.search(q):
        out.append(Period(raw="next year", fiscal_year=str(ref_year + 1)))

    return out


def _already_have_year(periods: List[Period], year: str) -> bool:
    return any(p.fiscal_year == year and not p.is_range for p in periods)


def _span_overlap(span: tuple, raw_str: str, full: str) -> bool:
    """True if `span` overlaps the substring `raw_str` inside `full`."""
    try:
        idx = full.lower().find(raw_str.lower())
        if idx < 0:
            return False
        end = idx + len(raw_str)
        s0, s1 = span
        return not (s1 <= idx or s0 >= end)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("What was Apple's revenue in FY2023?",                   ["2023"]),
        ("What was 3M's gross margin in fiscal year 2022?",        ["2022"]),
        ("Q4 FY2023 revenue for Apple?",                           ["2023"]),
        ("Between FY2021 and FY2022, how did revenue grow?",       ["2021", "2022"]),
        ("Q1 2023 EPS?",                                            ["2023"]),
        ("What is the trailing twelve months revenue?",             []),     # only TTM flag
        ("What was the year-end FY2018 net PPNE?",                 ["2018"]),
        ("Revenue in 2022 vs 2021",                                 ["2022", "2021"]),
        ("What about FY'22?",                                       ["2022"]),
        ("Compare last year to this year",                          [str(date.today().year-1),
                                                                     str(date.today().year)]),
    ]

    print("period_extractor — self test")
    passed = 0
    for q, expected_years in cases:
        ps = extract_periods(q)
        years = [p.fiscal_year for p in ps if p.fiscal_year and not p.is_ttm]
        ok = sorted(years) == sorted(expected_years)
        if ok:
            passed += 1
        flags = []
        if any(p.is_ttm for p in ps):       flags.append("TTM")
        if any(p.is_year_end for p in ps):  flags.append("YE")
        if any(p.is_range for p in ps):     flags.append("RANGE")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"  [{'✓' if ok else '✗'}] expect={expected_years!r:<22} "
              f"got={years!r:<22}{flag_str}  | {q[:50]}")
    print(f"\n  {passed}/{len(cases)} passed")
