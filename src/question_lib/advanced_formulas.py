"""
question_lib/advanced_formulas.py

ADVANCED MULTI-YEAR FORMULA SOLVER  (2026-06-21)

The DAG executor handles single-year ratios. FinanceBench's hardest SILVER
tier needs MULTI-YEAR AVERAGING and multi-operand formulas:

  - ROA           = NI(t) / avg(total_assets[t], total_assets[t-1])
  - fixed_asset_turnover = revenue(t) / avg(ppe[t], ppe[t-1])
  - DPO           = 365 * avg(AP[t], AP[t-1]) / (COGS(t) + (inv[t] - inv[t-1]))
  - 3yr avg capex%= mean( capex[y]/revenue[y] for y in last 3 )
  - inventory_turnover = COGS(t) / inventory(t)    (single year, but here for completeness)
  - YoY growth    = (x[t] - x[t-1]) / x[t-1] * 100
  - quick_ratio   = (current_assets - inventory) / current_liabilities

This solver reads the normalized (metric, year) -> value map built by
table_normalizer and computes the answer directly. Deterministic. No LLM.
Returns (value, unit) or None.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _year_int(period: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(period or ""))
    return int(m.group(0)) if m else None


def _get(norm: Dict, metric: str, year: int) -> Optional[float]:
    return norm.get((metric, str(year)))


def _avg2(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


_NUM_RE = re.compile(r"\(?\$?\s*-?\s*\d[\d,]*(?:\.\d+)?\)?")


# Raw-text fallback: some 10-Ks don't expose the income-statement /
# balance-sheet line as a structured table row (the PDF extractor missed it).
# When an operand is absent from the normalized map we do a SCOPED raw_text
# scan for that exact metric+year. Conservative: requires the metric phrase,
# rejects anti-patterns, and only returns a value in a sane magnitude band.
_RAW_SYNS = {
    "operating_income": (["operating income", "income from operations",
                          "operating profit", "operating earnings"],
                         ["non-operating", "nonoperating", "non operating"]),
    "ppe": (["property and equipment, net", "property, plant and equipment, net",
             "property plant and equipment net", "net property and equipment",
             "property and equipment net"],
            ["gross", "accumulated depreciation"]),
    "cogs": (["cost of sales", "cost of goods sold", "cost of products sold",
              "total cost of sales", "cost of revenue", "cost of revenues"],
             ["percentage", "% of", "as a %"]),
    "accounts_payable": (["accounts payable"],
                         ["days", "turnover", "increase", "decrease",
                          "changes in", "change in"]),
    "inventory": (["inventories", "inventory"],
                  ["turnover", "days", "obsolete", "changes in",
                   "change in", "decrease in", "increase in",
                   "raw materials", "work in process", "finished goods",
                   "held for sale"]),
}

# Metrics for which the normalized-map value is UNRELIABLE (the table
# extractor often grabs a cash-flow 'change in X' line or a sub-component
# instead of the balance-sheet total). For these we take the balance-sheet
# line from raw_text, mapped to the right year.
#   Amazon AP: map had cash-flow 7,175; real balance-sheet AP 34,616.
#   Amazon inventory: map had sub-line 3,583; real total inventory 16,047.
#   Activision PP&E: _raw_lookup returned 253 for BOTH years; the real
#     balance-sheet line has 253 (2019) and 282 (2018) - needs year-mapping.
_PREFER_RAW_MAX = {"accounts_payable", "inventory", "ppe"}


def _raw_lookup_max(raw_text: str, metric: str, year: int) -> Optional[float]:
    """Find the balance-sheet line for *metric* and return the value for the
    requested *year*, mapped POSITIONALLY (balance sheets print 2 columns:
    newest and prior year). We pick the occurrence whose line has the LARGEST
    values (the real balance-sheet total dominates a cash-flow 'change in X'
    line), then map its numbers to years.

    Returns the year-correct value, NOT a blanket max (which would give every
    year the same number). Scoped to metrics in _RAW_SYNS.
    """
    if not raw_text or metric not in _RAW_SYNS:
        return None
    positives, antis = _RAW_SYNS[metric]
    low = raw_text.lower()

    # Collect every candidate line's numeric run + the header context that
    # precedes it (used later to detect column year-order).
    candidate_runs = []  # list of (floats, header_text)
    for phrase in positives:
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx < 0:
                break
            start = idx + len(phrase)
            line_end = raw_text.find("\n", idx)
            if line_end < 0:
                line_end = idx + 160
            window = raw_text[idx:line_end]
            wlow = window.lower()
            if any(a in wlow for a in antis):
                continue
            # header context = up to 400 chars before the line (column years
            # like 'December 31, 2017 2016' live just above the data rows).
            header_ctx = raw_text[max(0, idx - 400): idx]
            nums = []
            for m in _NUM_RE.finditer(window):
                cleaned = re.sub(r"[\$\s,()]", "", m.group(0))
                try:
                    v = abs(float(cleaned))
                except ValueError:
                    continue
                if 1990 <= v <= 2035 and v == int(v):
                    continue
                if v >= 1_000_000:
                    v /= 1000.0
                if v > 0:
                    nums.append(v)
            if nums:
                candidate_runs.append((nums, header_ctx))

    if not candidate_runs:
        return None

    # Prefer a line that has at least 2 values (a real balance-sheet line
    # prints BOTH the current and prior year, e.g. '25,309  34,616'). Among
    # those, pick the one whose max value is largest (the real total beats a
    # small cash-flow 'change in AP' line).
    multi = [(r, hdr) for (r, hdr) in candidate_runs if len(r) >= 2]
    pool = multi if multi else candidate_runs
    best_run, best_hdr = max(pool, key=lambda rh: max(rh[0]))

    # Determine column order from the YEAR TOKENS in the header text that
    # precedes this line (10-K column headers like 'December 31, 2017 2016').
    # If the years there are descending (2017, 2016) the run is newest-first;
    # if ascending (2016, 2017) it's oldest-first and we reverse. This works
    # for BOTH growing metrics (AP 25309->34616 oldest-first) and shrinking
    # ones (PPE 253<-282), instead of a fragile magnitude assumption.
    run = list(best_run)
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", best_hdr or "")]
    yrs = [y for y in yrs if 1995 <= y <= 2035]
    if len(yrs) >= 2:
        if yrs[0] < yrs[1]:          # header is oldest-first -> reverse data
            run = list(reversed(run))
    # else: no header years found -> trust document order (newest-first)
    return run


def _raw_lookup(raw_text: str, metric: str, year: int) -> Optional[float]:
    """Scoped raw_text scan for one metric+year. Returns millions or None."""
    if not raw_text or metric not in _RAW_SYNS:
        return None
    positives, antis = _RAW_SYNS[metric]
    low = raw_text.lower()
    ytok = str(year)
    for phrase in positives:
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx < 0:
                break
            start = idx + len(phrase)
            # window after the phrase, on the same logical line
            window = raw_text[idx: idx + 240]
            wlow = window.lower()
            # reject anti-pattern context (e.g. non-operating income)
            pre = low[max(0, idx - 20): idx]
            if any(a in pre or a in wlow[:len(phrase) + 4] for a in antis):
                continue
            # require the year to be near (income statements show 3 yrs in a row)
            if ytok not in window and ytok not in low[max(0, idx - 60): idx]:
                # still allow if it's the only/first occurrence with numbers
                pass
            nums = []
            for m in _NUM_RE.finditer(window):
                tok = m.group(0)
                neg = "(" in tok and ")" in tok
                cleaned = re.sub(r"[\$\s,()]", "", tok)
                try:
                    v = float(cleaned)
                except ValueError:
                    continue
                if neg:
                    v = -v
                # skip year-like ints
                if 1990 <= abs(v) <= 2035 and v == int(v):
                    continue
                nums.append(v)
            if nums:
                val = nums[0]
                # normalize thousands -> millions
                if abs(val) >= 1_000_000:
                    val /= 1000.0
                return val
    return None


def _operand(norm: Dict, raw_text: str, metric: str, year: int,
             anchor_year: Optional[int] = None) -> Optional[float]:
    """Get an operand from the normalized map, falling back to raw_text.

    For metrics in _PREFER_RAW_MAX the map value is known-unreliable (the
    extractor grabs a cash-flow 'change in X' line), so we take the
    balance-sheet line from raw_text and map it to the requested year by
    offset from the newest reporting year.
    """
    v = norm.get((metric, str(year)))
    if metric in _PREFER_RAW_MAX:
        run = _raw_lookup_max(raw_text, metric, year)
        if run:
            # run is newest-first; offset 0 = anchor (latest FY), 1 = prior...
            newest = anchor_year if anchor_year is not None else year
            offset = newest - year
            if 0 <= offset < len(run):
                raw_v = run[offset]
                # only override the map value if the raw balance-sheet value
                # is materially larger (the map had the small cash-flow line)
                if v is None or raw_v > abs(v):
                    return raw_v
        return v
    if v is not None:
        return v
    return _raw_lookup(raw_text, metric, year)


# ---------------------------------------------------------------------------
# Question-pattern detection
# ---------------------------------------------------------------------------

def detect_advanced(question: str) -> Optional[str]:
    """Return an advanced-formula id if the question matches one, else None."""
    q = (question or "").lower()

    # DPO - days payable outstanding
    if "days payable" in q or "dpo" in q:
        return "dpo"
    # DSO - days sales outstanding
    if "days sales outstanding" in q or "dso" in q:
        return "dso"
    # DIO - days inventory outstanding
    if "days inventory" in q or "dio" in q:
        return "dio"
    # ROA with averaging
    if "return on assets" in q or "roa" in q:
        return "roa_avg"
    # ROE with averaging
    if "return on equity" in q or ("roe" in q and "average" in q):
        return "roe_avg"
    # Fixed asset turnover (avg PP&E)
    if "fixed asset turnover" in q:
        return "fixed_asset_turnover"
    # Asset turnover with averaging
    if "asset turnover" in q and "average" in q:
        return "asset_turnover_avg"
    # Inventory turnover
    if "inventory" in q and ("turnover" in q or "sold its inventory" in q
                             or "times" in q):
        return "inventory_turnover"
    # 3-year average capex as % of revenue
    if "capex" in q and "average" in q and "%" in q:
        return "capex_pct_3yr_avg"
    if "capex as a %" in q or "capex as a percent" in q:
        return "capex_pct_3yr_avg" if "average" in q else "capex_pct"
    # YoY change / growth of a named metric
    if ("year-over-year change" in q or "year over year change" in q
            or "yoy change" in q):
        return "yoy_change"
    # quick ratio
    if "quick ratio" in q:
        return "quick_ratio_2yr"

    return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(question: str, norm: Dict, fiscal_year: str,
          raw_text: str = "") -> Optional[Tuple[float, str]]:
    """Compute an advanced formula from the normalized map.

    raw_text enables a scoped fallback for operands (operating_income, ppe,
    cogs) that the PDF table extractor missed. Returns (value, unit) or None.
    """
    fid = detect_advanced(question)
    if not fid:
        return None
    t = _year_int(fiscal_year)
    if t is None:
        # try to read year from the question itself
        ys = [int(y) for y in re.findall(r"(?:19|20)\d{2}", question or "")]
        t = max(ys) if ys else None
    if t is None:
        return None
    p = t - 1  # prior year

    try:
        if fid == "roa_avg":
            ni = _get(norm, "net_income", t)
            avg_assets = _avg2(_get(norm, "total_assets", t),
                               _get(norm, "total_assets", p))
            if ni is not None and avg_assets:
                return (ni / avg_assets, "ratio")

        if fid == "roe_avg":
            ni = _get(norm, "net_income", t)
            avg_eq = _avg2(_get(norm, "shareholders_equity", t),
                           _get(norm, "shareholders_equity", p))
            if ni is not None and avg_eq:
                return (ni / avg_eq, "ratio")

        if fid == "fixed_asset_turnover":
            rev = _get(norm, "revenue", t)
            avg_ppe = _avg2(_operand(norm, raw_text, "ppe", t, anchor_year=t),
                            _operand(norm, raw_text, "ppe", p, anchor_year=t))
            if rev is not None and avg_ppe:
                return (rev / avg_ppe, "x")
        if fid == "asset_turnover_avg":
            rev = _get(norm, "revenue", t)
            avg_assets = _avg2(_get(norm, "total_assets", t),
                               _get(norm, "total_assets", p))
            if rev is not None and avg_assets:
                return (rev / avg_assets, "x")

        if fid == "inventory_turnover":
            cogs = _operand(norm, raw_text, "cogs", t)
            inv = _get(norm, "inventory", t)
            if cogs is not None and inv:
                return (cogs / inv, "x")

        if fid == "dpo":
            avg_ap = _avg2(
                _operand(norm, raw_text, "accounts_payable", t, anchor_year=t),
                _operand(norm, raw_text, "accounts_payable", p, anchor_year=t))
            cogs = _operand(norm, raw_text, "cogs", t)
            inv_t = _operand(norm, raw_text, "inventory", t, anchor_year=t)
            inv_p = _operand(norm, raw_text, "inventory", p, anchor_year=t)
            if avg_ap and cogs is not None and inv_t is not None and inv_p is not None:
                denom = cogs + (inv_t - inv_p)
                if abs(denom) > 1e-9:
                    return (365.0 * avg_ap / denom, "days")

        if fid == "dio":
            avg_inv = _avg2(_get(norm, "inventory", t), _get(norm, "inventory", p))
            cogs = _get(norm, "cogs", t)
            if avg_inv and cogs:
                return (365.0 * avg_inv / cogs, "days")

        if fid == "dso":
            avg_ar = _avg2(_get(norm, "accounts_receivable", t),
                           _get(norm, "accounts_receivable", p))
            rev = _get(norm, "revenue", t)
            if avg_ar and rev:
                return (365.0 * avg_ar / rev, "days")

        if fid == "capex_pct_3yr_avg":
            ratios = []
            for y in (t, t - 1, t - 2):
                cx = _get(norm, "capex", y)
                rv = _get(norm, "revenue", y)
                if cx is not None and rv:
                    ratios.append(abs(cx) / rv * 100.0)
            if len(ratios) >= 2:
                return (sum(ratios) / len(ratios), "%")

        if fid == "capex_pct":
            cx = _get(norm, "capex", t)
            rv = _get(norm, "revenue", t)
            if cx is not None and rv:
                return (abs(cx) / rv * 100.0, "%")

        if fid == "quick_ratio_2yr":
            ca = _get(norm, "current_assets", t)
            inv = _get(norm, "inventory", t)
            cl = _get(norm, "current_liabilities", t)
            if ca is not None and inv is not None and cl:
                qr = (ca - inv) / cl
                # sanity: a real quick ratio sits roughly in [0.1, 5]. A value
                # far outside that means an operand (often inventory or CL)
                # was mis-extracted -> abstain rather than emit garbage (the
                # Amcor 0.03 case came from a bad inventory/CL operand).
                if 0.1 <= qr <= 5.0:
                    return (qr, "x")
                return None

        if fid == "yoy_change":
            # figure out which metric the question is about
            metric = _detect_metric_in_question(question)
            if metric:
                cur = _operand(norm, raw_text, metric, t)
                prev = _operand(norm, raw_text, metric, p)
                # period-collapse guard: if the fallback returned the SAME
                # number for both years, the raw_text scan grabbed one shared
                # occurrence -> abstain rather than report a false 0.0%.
                if (cur is not None and prev is not None
                        and abs(cur - prev) < 1e-9):
                    return None
                if cur is not None and prev and abs(prev) > 1e-9:
                    return ((cur - prev) / abs(prev) * 100.0, "%")

    except Exception:
        logger.debug("[advanced_formulas] solve failed for %s", fid, exc_info=True)
        return None
    return None


def _detect_metric_in_question(question: str) -> Optional[str]:
    """Map words in a YoY-change question to a metric_id."""
    q = (question or "").lower()
    table = [
        ("operating income", "operating_income"),
        ("operating profit", "operating_income"),
        ("net income", "net_income"),
        ("net earnings", "net_income"),
        ("revenue", "revenue"),
        ("net sales", "revenue"),
        ("total sales", "revenue"),
        ("gross profit", "gross_profit"),
        ("cost of goods", "cogs"),
        ("cost of sales", "cogs"),
        ("total assets", "total_assets"),
        ("ebitda", "ebitda"),
        ("capital expenditure", "capex"),
        ("capex", "capex"),
    ]
    for phrase, mid in table:
        if phrase in q:
            return mid
    return None


def format_answer(value: float, unit: str) -> str:
    """Format an advanced-formula result to match FinanceBench style."""
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "days":
        return f"{value:.2f}"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "x":
        return f"{value:.2f}"
    return f"{value:.2f}"


if __name__ == "__main__":
    # synthetic self-test
    norm = {
        ("net_income", "2022"): -546.0, ("total_assets", "2022"): 38363.0,
        ("total_assets", "2021"): 32963.0,
        ("revenue", "2019"): 6489.0, ("ppe", "2019"): 253.0, ("ppe", "2018"): 282.0,
        ("revenue", "2017"): 177866.0, ("revenue", "2016"): 135987.0,
        ("accounts_payable", "2017"): 34616.0, ("accounts_payable", "2016"): 25309.0,
        ("cogs", "2017"): 111934.0, ("inventory", "2017"): 16047.0,
        ("inventory", "2016"): 11461.0,
        ("operating_income", "2016"): 1493.602, ("operating_income", "2015"): 903.095,
    }
    tests = [
        ("AES FY2022 return on assets (ROA)", "FY2022"),
        ("Activision FY2019 fixed asset turnover ratio", "FY2019"),
        ("Amazon FY2017 days payable outstanding DPO", "FY2017"),
        ("Amazon year-over-year change in revenue FY2016 to FY2017", "FY2017"),
        ("Adobe year-over-year change in operating income FY2015 to FY2016", "FY2016"),
    ]
    for q, fy in tests:
        r = solve(q, norm, fy)
        print(f"{q[:55]:55} -> {r}")
