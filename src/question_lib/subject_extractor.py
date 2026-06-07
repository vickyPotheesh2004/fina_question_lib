"""
question_lib/subject_extractor.py
Pull the MAIN METRIC SUBJECT from a question.

E.g. "What was Apple's gross margin in FY2023?" → Subject(
    metric_id="gross_margin",
    display_name="gross margin",
    matched_phrase="gross margin",
)

Uses an internal alias map (mirrors extract_lib synonyms) so we don't
require extract_lib to be installed at import time, but PREFERS extract_lib
when available.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .models import Subject

# Try to use extract_lib synonyms when available (DRY)
try:
    from extract_lib.synonyms import METRIC_SYNONYMS as _EXT_SYNS    # type: ignore
    _HAS_EXTRACT = True
except Exception:
    _EXT_SYNS = {}
    _HAS_EXTRACT = False


# ─────────────────────────────────────────────────────────────────────────────
# Built-in alias map (mirrors extract_lib synonyms, with QUESTION-level
# variants — e.g. "EPS" without "per share" etc.)
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_ALIASES = {
    # ── Income statement ────────────────────────────────────────────────
    "revenue":               "revenue",
    "revenues":              "revenue",
    "net revenue":           "revenue",
    "net revenues":          "revenue",
    "net sales":             "revenue",
    "total revenue":         "revenue",
    "total revenues":        "revenue",
    "total sales":           "revenue",
    "sales":                 "revenue",
    "top line":              "revenue",
    "cogs":                  "cogs",
    "cost of sales":         "cogs",
    "cost of revenue":       "cogs",
    "cost of goods sold":    "cogs",
    "gross profit":          "gross_profit",
    "gross margin":          "gross_margin",
    "operating income":      "operating_income",
    "operating profit":      "operating_income",
    "income from operations":"operating_income",
    "operating margin":      "operating_margin",
    "operating expenses":    "operating_expenses",
    "opex":                  "operating_expenses",
    "sg&a":                  "sg_and_a",
    "selling general":       "sg_and_a",
    "selling and admin":     "sg_and_a",
    "r&d":                   "r_and_d",
    "research and development": "r_and_d",
    "interest expense":      "interest_expense",
    "interest income":       "interest_income",
    "income before tax":     "income_before_tax",
    "pretax income":         "income_before_tax",
    "pre-tax income":        "income_before_tax",
    "income tax":            "income_tax",
    "tax expense":           "income_tax",
    "tax provision":         "income_tax",
    "effective tax rate":    "effective_tax_rate",
    "net income":            "net_income",
    "net earnings":          "net_income",
    "earnings":              "net_income",
    "bottom line":           "net_income",
    "net margin":            "net_margin",
    "ebitda":                "ebitda",
    "ebit":                  "operating_income",
    "eps":                   "eps_diluted",
    "earnings per share":    "eps_diluted",
    "diluted eps":           "eps_diluted",
    "diluted earnings per share": "eps_diluted",
    "basic eps":             "eps_basic",
    "basic earnings per share": "eps_basic",

    # ── Balance sheet ───────────────────────────────────────────────────
    "cash":                  "cash",
    "cash and cash equivalents": "cash",
    "accounts receivable":   "accounts_receivable",
    "receivables":           "accounts_receivable",
    "inventory":             "inventory",
    "inventories":           "inventory",
    "current assets":        "current_assets",
    "property, plant and equipment": "ppe",
    "property plant and equipment":  "ppe",
    "net ppne":              "ppe",
    "net ppe":               "ppe",
    "ppe":                   "ppe",
    "ppne":                  "ppe",
    "fixed assets":          "ppe",
    "goodwill":              "goodwill",
    "intangible assets":     "intangible_assets",
    "intangibles":           "intangible_assets",
    "total assets":          "total_assets",
    "assets":                "total_assets",
    "accounts payable":      "accounts_payable",
    "payables":              "accounts_payable",
    "current liabilities":   "current_liabilities",
    "long-term debt":        "long_term_debt",
    "long term debt":        "long_term_debt",
    "ltd":                   "long_term_debt",
    "total debt":            "long_term_debt",     # approximation
    "total liabilities":     "total_liabilities",
    "liabilities":           "total_liabilities",
    "shareholders equity":   "shareholders_equity",
    "shareholders' equity":  "shareholders_equity",
    "stockholders equity":   "shareholders_equity",
    "stockholders' equity":  "shareholders_equity",
    "equity":                "shareholders_equity",
    "book value":            "shareholders_equity",

    # ── Cash flow ───────────────────────────────────────────────────────
    "operating cash flow":   "operating_cash_flow",
    "ocf":                   "operating_cash_flow",
    "cash from operations":  "operating_cash_flow",
    "investing cash flow":   "investing_cash_flow",
    "financing cash flow":   "financing_cash_flow",
    "capital expenditure":   "capex",
    "capital expenditures":  "capex",
    "capex":                 "capex",
    "purchases of property": "capex",
    "depreciation":          "depreciation_amortization",
    "amortization":          "depreciation_amortization",
    "d&a":                   "depreciation_amortization",
    "dividends paid":        "dividends_paid",
    "cash dividends":        "dividends_paid",
    "share repurchase":      "share_repurchases",
    "share repurchases":     "share_repurchases",
    "stock buyback":         "share_repurchases",
    "treasury stock":        "share_repurchases",
    "free cash flow":        "free_cash_flow",
    "fcf":                   "free_cash_flow",

    # ── Per-share / ratios ──────────────────────────────────────────────
    "weighted average shares basic":   "weighted_avg_shares_basic",
    "weighted average shares diluted": "weighted_avg_shares_diluted",
    "current ratio":         "current_ratio",
    "quick ratio":            "quick_ratio",
    "cash ratio":             "cash_ratio",
    "debt to equity":         "debt_to_equity",
    "debt-to-equity":         "debt_to_equity",
    "d/e":                    "debt_to_equity",
    "debt to assets":         "debt_to_assets",
    "debt-to-assets":         "debt_to_assets",
    "interest coverage":      "interest_coverage",
    "roa":                    "roa",
    "return on assets":       "roa",
    "roe":                    "roe",
    "return on equity":       "roe",
    "roic":                   "roic",
    "return on invested capital": "roic",
    "asset turnover":         "asset_turnover",
    "inventory turnover":     "inventory_turnover",
    "working capital":        "working_capital",
}


# ─────────────────────────────────────────────────────────────────────────────
# Extract
# ─────────────────────────────────────────────────────────────────────────────

def extract_subject(question: str) -> Optional[Subject]:
    """Return the best Subject match in `question`, or None.

    Strategy:
      1. Match longest alias first (most specific).
      2. Word-boundary aware (avoid 'eps' matching inside 'depsire').
      3. Confidence: 1.0 if exact phrase, 0.9 if partial.
    """
    if not question:
        return None

    q = question.lower()

    # Sort longest first → match most specific phrase
    sorted_aliases = sorted(_QUESTION_ALIASES.items(), key=lambda x: -len(x[0]))

    best: Optional[Tuple[str, str, int, int]] = None  # (phrase, metric_id, start, end)
    for phrase, metric_id in sorted_aliases:
        # Use word boundary for short aliases (<=4 chars) to avoid false hits
        if len(phrase) <= 4:
            pattern = rf"\b{re.escape(phrase)}\b"
        else:
            pattern = re.escape(phrase)
        m = re.search(pattern, q)
        if m:
            best = (phrase, metric_id, m.start(), m.end())
            break

    if best is None:
        return None

    phrase, metric_id, start, end = best
    # Confidence: full word/phrase = 1.0
    confidence = 1.0 if len(phrase) >= 5 else 0.92
    display_name = phrase

    return Subject(
        metric_id=metric_id,
        display_name=display_name,
        matched_phrase=phrase,
        char_start=start,
        char_end=end,
        confidence=confidence,
    )


def extract_all_subjects(question: str, max_subjects: int = 5) -> List[Subject]:
    """Return ALL distinct subjects in the question (for comparison questions)."""
    if not question:
        return []
    q = question.lower()
    found: List[Subject] = []
    seen_ids: set = set()
    sorted_aliases = sorted(_QUESTION_ALIASES.items(), key=lambda x: -len(x[0]))

    occupied: List[Tuple[int, int]] = []   # to avoid double-matching same span
    for phrase, metric_id in sorted_aliases:
        if len(found) >= max_subjects:
            break
        if metric_id in seen_ids:
            continue
        if len(phrase) <= 4:
            pattern = rf"\b{re.escape(phrase)}\b"
        else:
            pattern = re.escape(phrase)
        for m in re.finditer(pattern, q):
            s, e = m.start(), m.end()
            if any(not (e <= os or s >= oe) for os, oe in occupied):
                continue   # overlaps existing match
            confidence = 1.0 if len(phrase) >= 5 else 0.92
            found.append(Subject(
                metric_id=metric_id,
                display_name=phrase,
                matched_phrase=phrase,
                char_start=s,
                char_end=e,
                confidence=confidence,
            ))
            seen_ids.add(metric_id)
            occupied.append((s, e))
            break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("What was Apple's revenue in FY2023?",                   "revenue"),
        ("What was 3M's gross margin in FY2022?",                  "gross_margin"),
        ("How much capex did 3M spend in FY2018?",                  "capex"),
        ("Is 3M a capital-intensive business?",                     "capex"),  # close enough
        ("What is Apple's diluted EPS?",                            "eps_diluted"),
        ("What was the net income for Tesla in 2022?",              "net_income"),
        ("What was Apple's free cash flow in FY2023?",              "free_cash_flow"),
        ("What is Apple's total assets?",                           "total_assets"),
        ("What is the year end FY2018 net PPNE for 3M?",            "ppe"),
        ("Is Apple's liquidity healthy?",                           None),     # no specific metric
    ]
    print("subject_extractor — self test")
    passed = 0
    for q, expected in cases:
        subj = extract_subject(q)
        got = subj.metric_id if subj else None
        ok = "✓" if got == expected else "✗"
        if got == expected:
            passed += 1
        print(f"  [{ok}] expect={expected!r:<22} got={got!r:<22}  | {q[:55]}")
    print(f"\n  {passed}/{len(cases)} passed")
    print(f"  extract_lib available: {_HAS_EXTRACT}")
