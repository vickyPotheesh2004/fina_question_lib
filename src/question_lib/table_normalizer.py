"""
question_lib/table_normalizer.py

POST-EXTRACTION NORMALIZATION LAYER  (2026-06-21)

Built once per document. Turns the messy structured_tables (year labels
scattered across headers / data rows / nowhere, units mixed thousands vs
millions, segment rows mixed with totals) into ONE clean lookup:

    normalized[(metric_id, "2019")] = 6489.0     # always in millions

Every (metric, year) question then becomes a dict lookup instead of a
per-query re-parse of raw tables. This is the root-cause fix for:
  - OCF ratio 0.00      (scale mismatch between operands)
  - YoY growth collapse (both years returning the same number)
  - wrong-column picks  (segment row chosen over statement total)

Deterministic. No LLM. Never raises.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from extract_lib.synonyms import METRIC_SYNONYMS as _EXT_SYNS
    _HAS_EXTRACT = True
except Exception:
    _EXT_SYNS = {}
    _HAS_EXTRACT = False


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[,;:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_number(token: str) -> Optional[float]:
    """Parse one cell into a float in MILLIONS-agnostic raw form.
    Returns the raw numeric value (scaling handled later)."""
    if token is None:
        return None
    raw = str(token).strip()
    if not raw:
        return None
    neg = "(" in raw and ")" in raw
    raw2 = re.sub(r"[\$\s\(\)%,]", "", raw)
    m = re.match(r"^([-\d\.]+)\s*(million|billion|thousand|bn|mn|m|b|k)?$",
                 raw2, re.IGNORECASE)
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
    elif suf in ("thousand", "k"):
        v /= 1000.0
    return v


def _is_year_token(s: str) -> bool:
    s = str(s).strip()
    if not re.fullmatch(r"(19|20)\d{2}", s):
        return False
    return 1995 <= int(s) <= 2035


def _looks_like_year_value(v: float) -> bool:
    if v is None:
        return False
    av = abs(v)
    return 1990 <= av <= 2035 and av == int(av)


# ---------------------------------------------------------------------------
# Year-axis detection per table
# ---------------------------------------------------------------------------

def _detect_year_columns(tbl: Dict) -> Optional[Dict[int, str]]:
    """If a table has a row of year labels, return {col_index: 'YYYY'}.

    Looks at headers first, then every data row, for a row containing >= 2
    distinct plausible years. Returns the column->year map for that row.
    """
    candidates = []
    headers = tbl.get("headers", []) or []
    rows = tbl.get("rows", []) or []
    all_rows = [headers] + rows if headers else rows

    for row in all_rows:
        if not row:
            continue
        col_year = {}
        for ci, c in enumerate(row):
            cs = str(c).strip()
            if _is_year_token(cs):
                col_year[ci] = cs
        distinct = set(col_year.values())
        if len(distinct) >= 2:
            candidates.append(col_year)

    if not candidates:
        return None
    # choose the row with the most year columns (the real fiscal header)
    best = max(candidates, key=lambda d: len(d))
    return best


def _document_year_order(structured_tables: List[Dict],
                         anchor_year: Optional[int] = None) -> List[str]:
    """Best-effort document fiscal-year ordering, NEWEST first.

    FIX (2026-06-21): constrain to the document's real fiscal range. 10-Ks
    contain MANY 4-digit numbers that are not reporting years (debt maturity
    2027, contractual-obligation 2024, history 2002). When we know the
    document's fiscal year (anchor_year), keep only years in
    [anchor_year - 6, anchor_year]; a financial statement never reports a
    year AFTER its fiscal year, and rarely more than 5 years back. This turns
    the Activision junk ['2027','2024',...,'2002'] into ['2019','2018','2017'].
    """
    from collections import Counter

    def _in_range(y: int) -> bool:
        if anchor_year is None:
            return 1995 <= y <= 2035
        return (anchor_year - 6) <= y <= anchor_year

    seen_sets = Counter()
    for tbl in structured_tables or []:
        cm = _detect_year_columns(tbl)
        if cm:
            ordered = [cm[k] for k in sorted(cm.keys()) if _in_range(int(cm[k]))]
            uniq = []
            for y in ordered:
                if y not in uniq:
                    uniq.append(y)
            if len(uniq) >= 2:
                seen_sets[tuple(uniq)] += 1

    if seen_sets:
        best = max(seen_sets.items(), key=lambda kv: (len(kv[0]), kv[1]))[0]
        nums = [int(y) for y in best]
        if nums == sorted(nums):
            best = tuple(reversed(best))
        result = list(best)
    else:
        result = []

    # FIX (2026-06-21): when we know the fiscal year, prefer the CANONICAL
    # consecutive run [anchor, anchor-1, anchor-2, anchor-3]. 10-K statements
    # always present the FY plus 2-4 consecutive prior years. Detected runs
    # often skip a year (Activision: 2019,2018,2016,2014 -- missing 2017),
    # which throws off positional mapping. If at least 3 of the canonical
    # years appear ANYWHERE in the doc, use the clean consecutive run.
    if anchor_year is not None:
        canonical = [anchor_year - i for i in range(5)]
        present = set()
        for tbl in structured_tables or []:
            for row in ([tbl.get("headers", [])] + (tbl.get("rows", []) or [])):
                for c in (row or []):
                    cs = str(c).strip()
                    if _is_year_token(cs):
                        present.add(int(cs))
        hits = [y for y in canonical if y in present]
        if len(hits) >= 3:
            return [str(y) for y in hits]

    if result:
        return result

    # Fallback: all distinct in-range years, newest-first.
    allyears = set()
    for tbl in structured_tables or []:
        for row in ([tbl.get("headers", [])] + (tbl.get("rows", []) or [])):
            for c in (row or []):
                cs = str(c).strip()
                if _is_year_token(cs) and _in_range(int(cs)):
                    allyears.add(int(cs))
    if len(allyears) >= 2:
        return [str(y) for y in sorted(allyears, reverse=True)]
    return []


# ---------------------------------------------------------------------------
# Metric row matching
# ---------------------------------------------------------------------------

def _row_metric(label: str) -> Optional[str]:
    """Map a row label to a metric_id using extract_lib synonyms + anti.
    Returns the metric_id whose positive synonym matches and no anti matches.
    Prefers the LONGEST matching positive (most specific).

    NOTE: anti-patterns are matched against the row LABEL only (the first
    cell), never against value cells. A '%' or 'ratio' appearing in a value
    column must not reject the metric row (this was making Adobe
    operating_income extract nothing).
    """
    if not _HAS_EXTRACT:
        return None
    nlabel = _norm(label)
    if not nlabel:
        return None
    # anti-patterns that are pure symbols (%/ratio) should not apply to the
    # label text itself unless the label literally is a percentage/ratio row.
    best_metric = None
    best_len = -1
    for metric_id, syn in _EXT_SYNS.items():
        positives = [_norm(s) for s in syn.get("positive", []) if s]
        raw_anti = [_norm(s) for s in (syn.get("anti") or syn.get("negative") or []) if s]
        # only word-like anti-patterns apply to a label; drop bare '%'/symbols
        negatives = [a for a in raw_anti if len(a) >= 3 and not a.isdigit()]
        if any(neg and neg in nlabel for neg in negatives):
            continue
        for p in positives:
            if p and p in nlabel and len(p) > best_len:
                best_metric = metric_id
                best_len = len(p)
    return best_metric


def _leading_value_run(row: List, keep_negatives: bool = False) -> List[float]:
    """Numbers from a row.

    Default: stop at the first negative (change / %-change columns are negative
    or follow the year values). Skips year-like values.

    keep_negatives=True: collect negatives as their ABSOLUTE value instead of
    stopping. Cost / expense rows in 10-Ks are often printed in parentheses
    e.g. 'Cost of sales (10,069) (9,123)' which parse as negative; stopping at
    the first negative dropped the ENTIRE row (AES COGS came out as a tiny
    504 sub-line instead of the ~10,000 total). For cost/expense metrics we
    keep the magnitudes so the real total is captured.
    """
    nums = []
    for c in row[1:]:
        v = _parse_number(str(c))
        if v is None or _looks_like_year_value(v):
            continue
        if v < 0:
            if keep_negatives:
                nums.append(abs(v))
                continue
            break
        nums.append(v)
    return nums


# ---------------------------------------------------------------------------
# Scale normalization to MILLIONS
# ---------------------------------------------------------------------------

def _normalize_scale(value: float, metric_id: str) -> float:
    """Force a value into MILLIONS.

    A single statement line item >= 1,000,000 in raw form is in THOUSANDS
    ($1T+ in millions is implausible for these metrics) -> /1000. Some 10-Ks
    report in whole dollars, so a balance-sheet line can come in at billions
    of raw units (Netflix current_liabilities 15,430,786,000 -> 15,430,786
    thousands -> 15,430 millions). We divide by 1000 repeatedly until the
    value is in a plausible millions band for a statement line item.
    Per-share / ratio metrics are left untouched.
    """
    if value is None:
        return value
    PER_SHARE = {"eps_basic", "eps_diluted", "effective_tax_rate"}
    if metric_id in PER_SHARE:
        return value
    v = float(value)
    # collapse thousands (and whole-dollar) reporting down to millions.
    # A single line item in millions is essentially never >= 1e7 ($10T);
    # if it is, the source was in thousands or dollars -> scale down.
    guard = 0
    while abs(v) >= 10_000_000.0 and guard < 4:
        v /= 1000.0
        guard += 1
    if abs(v) >= 1_000_000.0:
        v /= 1000.0
    return v


# ---------------------------------------------------------------------------
# Public: build the normalized map
# ---------------------------------------------------------------------------

def build_normalized(structured_tables: List[Dict],
                     doc_fiscal_year: Optional[str] = None) -> Dict[Tuple[str, str], float]:
    """Return {(metric_id, 'YYYY'): value_in_millions}.

    doc_fiscal_year (e.g. 'FY2019' or '2019') anchors year detection to the
    real fiscal range, filtering out debt-maturity / contractual junk years.
    """
    out: Dict[Tuple[str, str], float] = {}
    is_total_src: Dict[Tuple[str, str], bool] = {}

    anchor = None
    if doc_fiscal_year:
        m = _YEAR_RE.search(str(doc_fiscal_year))
        if m:
            anchor = int(m.group(0))

    doc_years = _document_year_order(structured_tables, anchor_year=anchor)

    for tbl in structured_tables or []:
        rows = list(tbl.get("rows", []) or [])
        headers = tbl.get("headers", []) or []
        all_rows = ([headers] + rows) if headers else rows

        col_year = _detect_year_columns(tbl)   # may be None

        for row in all_rows:
            if not row:
                continue
            label = str(row[0]) if row else ""
            metric_id = _row_metric(label)
            if not metric_id:
                continue
            nlabel = _norm(label)
            is_total = nlabel.startswith("total") or nlabel.startswith("consolidated")

            # cost/expense rows are often parenthesized (negative) - keep
            # magnitudes so the real total isn't dropped. Applies to BOTH
            # the year-column path (A) and the positional path (B).
            _expense = metric_id in {
                "cogs", "operating_expenses", "sg_and_a", "r_and_d",
                "interest_expense", "income_tax", "capex",
                "depreciation_amortization",
            }

            if col_year:
                # Path A: explicit year columns
                for ci, yr in col_year.items():
                    if ci < len(row):
                        v = _parse_number(str(row[ci]))
                        if v is None or _looks_like_year_value(v):
                            continue
                        # expense rows: use absolute magnitude
                        if _expense and v < 0:
                            v = abs(v)
                        _store(out, is_total_src, metric_id, yr,
                               _normalize_scale(v, metric_id), is_total)
            else:
                # Path B: positional mapping via document year order
                if not doc_years:
                    continue
                nums = _leading_value_run(row, keep_negatives=_expense)
                # If the row is ascending (oldest first) flip to match
                # doc_years which is newest-first.
                vals = list(nums)
                if len(vals) >= 2 and vals[0] < vals[-1]:
                    vals = list(reversed(vals))
                for idx, yr in enumerate(doc_years):
                    if idx < len(vals):
                        _store(out, is_total_src, metric_id, yr,
                               _normalize_scale(vals[idx], metric_id), is_total)

    logger.debug("[table_normalizer] built %d (metric,year) entries", len(out))
    return out


# Metrics where a tiny WRONG sub-row competes with the real line and the
# real line is the LARGER absolute value. COGS matched tiny
# 'cost of services' sub-lines; accounts_payable matched the small cash-flow
# 'change in AP' instead of the balance-sheet total. For these, magnitude-wins.
#
# Deliberately EXCLUDED: current_assets / current_liabilities / total_assets /
# total_liabilities. Those have multiple legitimate 'Total ...' rows across
# periods and statements where the LARGER one is often the wrong period or a
# held-for-sale-inclusive figure (Amcor CL: real 4,393 < distractor 5,103).
# For those we rely on first-write + total-priority instead.
_MAGNITUDE_WINS = {
    "cogs", "accounts_payable",
}


def _store(out, is_total_src, metric_id, year, value, is_total):
    key = (metric_id, year)
    if key not in out:
        out[key] = value
        is_total_src[key] = is_total
        return
    # A 'total'/'consolidated' row always overrides a non-total row.
    if is_total and not is_total_src.get(key, False):
        out[key] = value
        is_total_src[key] = True
        return
    # For statement-total metrics, the largest matching value is the real
    # total (kills tiny wrong sub-rows like Amazon COGS=67, and the
    # cash-flow 'change in AP' 7,175 vs the balance-sheet AP 34,616).
    # Magnitude-wins applies regardless of total-ness tier here because the
    # real balance-sheet line is simply the larger absolute value for these
    # metrics, whether or not its label starts with 'total'.
    if metric_id in _MAGNITUDE_WINS:
        if abs(value) > abs(out[key]):
            out[key] = value
            is_total_src[key] = is_total
        return
    # Otherwise KEEP the first write. The old blanket 'larger magnitude wins'
    # rule let a stray bigger number from a DIFFERENT row overwrite the
    # correct value (Activision revenue@2019 6489 overwritten by 187890).
    # First correct write stays for small/balance-sheet metrics.


def lookup(normalized: Dict[Tuple[str, str], float],
           metric_id: str, period: str) -> Optional[float]:
    """Look up (metric, year) from a prebuilt normalized map."""
    if not normalized:
        return None
    ym = _YEAR_RE.search(period or "")
    if not ym:
        return None
    return normalized.get((metric_id, ym.group(0)))


if __name__ == "__main__":
    # tiny self-test with synthetic tables mimicking Activision + Amazon
    activision = [
        {"headers": ["Consolidated net revenues", "$", "6,489", "$", "7,500",
                     "$", "(1,011", ")", "(13", ")%"],
         "rows": []},
    ]
    amazon = [
        {"headers": ["", "2017", "2016", "2015"], "rows": [
            ["Total net sales", "177,866", "135,987", "107,006"],
        ]},
    ]
    print("Activision:", build_normalized(activision))
    print("Amazon:    ", build_normalized(amazon))
