"""
question_lib/formula_matcher.py
Map (subject, operation) → maths_lib formula signature.

E.g.
  Subject(metric_id="gross_margin"), Operation.RATIO_PCT
       → FormulaMatch(formula_id="gross_margin",
                       inputs=["gross_profit", "revenue"],
                       output_unit="%",
                       maths_lib_fn="gross_margin")

Internally maintains a static registry of ~50 common financial formulas.
Falls back gracefully when maths_lib is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Operation

# Try maths_lib if installed (preferred source of truth)
try:
    import maths_lib  # type: ignore
    _HAS_MATHS = True
except Exception:
    maths_lib = None
    _HAS_MATHS = False


# ─────────────────────────────────────────────────────────────────────────────
# Static formula registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormulaMatch:
    formula_id:    str            # e.g. "gross_margin"
    inputs:        List[str]      # extract_lib metric ids the formula needs
    output_unit:   str            # "%", "x", "$M", "days", ""
    operation:    Operation
    maths_lib_fn: str = ""         # maths_lib function name if known
    description:  str = ""
    confidence:   float = 1.0


# Map: (metric_id, Operation) → FormulaMatch
# When operation is None we use the default for that metric.
_REGISTRY: Dict[str, FormulaMatch] = {

    # ── PROFITABILITY ──────────────────────────────────────────────────
    "gross_margin": FormulaMatch(
        formula_id="gross_margin",
        inputs=["gross_profit", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="gross_margin",
        description="(gross_profit / revenue) * 100",
    ),
    "operating_margin": FormulaMatch(
        formula_id="operating_margin",
        inputs=["operating_income", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="operating_margin",
        description="(operating_income / revenue) * 100",
    ),
    "net_margin": FormulaMatch(
        formula_id="net_margin",
        inputs=["net_income", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="net_margin",
        description="(net_income / revenue) * 100",
    ),
    "ebitda_margin": FormulaMatch(
        formula_id="ebitda_margin",
        inputs=["ebitda", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="ebitda_margin",
        description="(ebitda / revenue) * 100",
    ),
    "fcf_margin": FormulaMatch(
        formula_id="fcf_margin",
        inputs=["free_cash_flow", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="fcf_margin",
        description="(fcf / revenue) * 100",
    ),
    "roa": FormulaMatch(
        formula_id="return_on_assets",
        inputs=["net_income", "total_assets"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="roa",
    ),
    "roe": FormulaMatch(
        formula_id="return_on_equity",
        inputs=["net_income", "shareholders_equity"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="roe",
    ),
    "roic": FormulaMatch(
        formula_id="return_on_invested_capital",
        inputs=["operating_income", "shareholders_equity", "long_term_debt"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="roic",
    ),

    # ── LIQUIDITY ──────────────────────────────────────────────────────
    "current_ratio": FormulaMatch(
        formula_id="current_ratio",
        inputs=["current_assets", "current_liabilities"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="current_ratio",
    ),
    "quick_ratio": FormulaMatch(
        formula_id="quick_ratio",
        inputs=["current_assets", "inventory", "current_liabilities"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="quick_ratio",
    ),
    "cash_ratio": FormulaMatch(
        formula_id="cash_ratio",
        inputs=["cash", "current_liabilities"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="cash_ratio",
    ),
    "working_capital": FormulaMatch(
        formula_id="working_capital",
        inputs=["current_assets", "current_liabilities"],
        output_unit="$M",
        operation=Operation.DIFF,
        maths_lib_fn="working_capital",
    ),

    # ── LEVERAGE ───────────────────────────────────────────────────────
    "debt_to_equity": FormulaMatch(
        formula_id="debt_to_equity",
        inputs=["long_term_debt", "shareholders_equity"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="debt_to_equity",
    ),
    "debt_to_assets": FormulaMatch(
        formula_id="debt_to_assets",
        inputs=["long_term_debt", "total_assets"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="debt_to_assets",
    ),
    "interest_coverage": FormulaMatch(
        formula_id="interest_coverage",
        inputs=["operating_income", "interest_expense"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="interest_coverage",
    ),

    # ── EFFICIENCY ─────────────────────────────────────────────────────
    "asset_turnover": FormulaMatch(
        formula_id="asset_turnover",
        inputs=["revenue", "total_assets"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="asset_turnover",
    ),
    "inventory_turnover": FormulaMatch(
        formula_id="inventory_turnover",
        inputs=["cogs", "inventory"],
        output_unit="x",
        operation=Operation.RATIO,
        maths_lib_fn="inventory_turnover",
    ),

    # ── CAPITAL INTENSITY ──────────────────────────────────────────────
    "capex_to_revenue": FormulaMatch(
        formula_id="capex_to_revenue",
        inputs=["capex", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="capex_intensity",
        description="(capex / revenue) * 100",
    ),
    "capital_intensity": FormulaMatch(
        formula_id="capital_intensity",
        inputs=["capex", "revenue"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="capex_intensity",
    ),

    # ── PAYOUT / DIVIDENDS ─────────────────────────────────────────────
    "payout_ratio": FormulaMatch(
        formula_id="payout_ratio",
        inputs=["dividends_paid", "net_income"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
        maths_lib_fn="payout_ratio",
    ),
    "retention_ratio": FormulaMatch(
        formula_id="retention_ratio",
        inputs=["dividends_paid", "net_income"],
        output_unit="%",
        operation=Operation.RATIO_PCT,
    ),

    # ── PER SHARE ──────────────────────────────────────────────────────
    "eps_diluted": FormulaMatch(
        formula_id="eps_diluted",
        inputs=["net_income", "weighted_avg_shares_diluted"],
        output_unit="$",
        operation=Operation.RATIO,
        maths_lib_fn="eps_diluted",
    ),
    "eps_basic": FormulaMatch(
        formula_id="eps_basic",
        inputs=["net_income", "weighted_avg_shares_basic"],
        output_unit="$",
        operation=Operation.RATIO,
        maths_lib_fn="eps_basic",
    ),
    "book_value_per_share": FormulaMatch(
        formula_id="book_value_per_share",
        inputs=["shareholders_equity", "weighted_avg_shares_diluted"],
        output_unit="$",
        operation=Operation.RATIO,
    ),

    # ── GROWTH ─────────────────────────────────────────────────────────
    "revenue_growth_yoy": FormulaMatch(
        formula_id="growth_yoy",
        inputs=["revenue_t", "revenue_t_minus_1"],
        output_unit="%",
        operation=Operation.GROWTH_YOY,
        maths_lib_fn="growth_yoy",
    ),
    "net_income_growth_yoy": FormulaMatch(
        formula_id="growth_yoy",
        inputs=["net_income_t", "net_income_t_minus_1"],
        output_unit="%",
        operation=Operation.GROWTH_YOY,
    ),

    # ── CASH FLOW ──────────────────────────────────────────────────────
    "free_cash_flow": FormulaMatch(
        formula_id="free_cash_flow",
        inputs=["operating_cash_flow", "capex"],
        output_unit="$M",
        operation=Operation.DIFF,
        maths_lib_fn="free_cash_flow",
    ),
}


# Aliases for variant subject IDs from subject_extractor
_ALIAS_TO_FORMULA = {
    "ppe":                   None,      # extraction only
    "capex":                 "capex_to_revenue",   # when computing capital intensity
    "revenue":               None,      # extraction only
    "net_income":            None,
    "cogs":                  None,
    "operating_income":      None,
    "gross_profit":          None,
    "gross_margin":          "gross_margin",
    "operating_margin":      "operating_margin",
    "net_margin":            "net_margin",
    "ebitda_margin":         "ebitda_margin",
    "ebitda":                None,      # extraction only
    "roa":                   "roa",
    "roe":                   "roe",
    "roic":                  "roic",
    "current_ratio":         "current_ratio",
    "quick_ratio":           "quick_ratio",
    "cash_ratio":            "cash_ratio",
    "debt_to_equity":        "debt_to_equity",
    "debt_to_assets":        "debt_to_assets",
    "interest_coverage":     "interest_coverage",
    "asset_turnover":        "asset_turnover",
    "inventory_turnover":    "inventory_turnover",
    "working_capital":       "working_capital",
    "free_cash_flow":        "free_cash_flow",
    "fcf":                   "free_cash_flow",
    "eps_diluted":           "eps_diluted",
    "eps_basic":             "eps_basic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def match_formula(
    metric_id: str,
    operation: Optional[Operation] = None,
) -> Optional[FormulaMatch]:
    """Return FormulaMatch for (metric_id, operation), or None.

    None means: "this metric is a direct extract, no formula needed".
    """
    if not metric_id:
        return None

    # Direct hit
    if metric_id in _REGISTRY:
        return _REGISTRY[metric_id]

    # Alias hit
    alias = _ALIAS_TO_FORMULA.get(metric_id)
    if alias and alias in _REGISTRY:
        return _REGISTRY[alias]

    return None


def list_supported_formulas() -> List[str]:
    """Return all formula_ids in the registry."""
    return sorted(_REGISTRY.keys())


def get_required_inputs(metric_id: str) -> List[str]:
    """Convenience: return list of inputs needed for the matched formula."""
    match = match_formula(metric_id)
    return list(match.inputs) if match else []


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("gross_margin",       ["gross_profit", "revenue"],          "%"),
        ("operating_margin",   ["operating_income", "revenue"],      "%"),
        ("current_ratio",      ["current_assets", "current_liabilities"], "x"),
        ("debt_to_equity",     ["long_term_debt", "shareholders_equity"], "x"),
        ("roe",                ["net_income", "shareholders_equity"],   "%"),
        ("free_cash_flow",     ["operating_cash_flow", "capex"],     "$M"),
        ("revenue",            None,                                 None),   # extract only
    ]
    print("formula_matcher — self test")
    passed = 0
    for metric_id, exp_inputs, exp_unit in cases:
        m = match_formula(metric_id)
        if exp_inputs is None:
            ok = m is None
        else:
            ok = m is not None and m.inputs == exp_inputs and m.output_unit == exp_unit
        if ok:
            passed += 1
        if m:
            print(f"  [{'✓' if ok else '✗'}] {metric_id:<20} → {m.formula_id:<22} "
                  f"inputs={m.inputs}  unit={m.output_unit}")
        else:
            print(f"  [{'✓' if ok else '✗'}] {metric_id:<20} → (no formula, extract-only)")
    print(f"\n  {passed}/{len(cases)} passed")
    print(f"  total formulas: {len(_REGISTRY)}")
    print(f"  maths_lib available: {_HAS_MATHS}")
