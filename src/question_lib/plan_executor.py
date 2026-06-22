"""
question_lib/plan_executor.py
Execute a QuestionPlan deterministically using the 7 support libraries.

Pipeline:
  1. Resolve each ExtractRequest    -> extract_lib.resolve_metric (or raw_text)
  2. Execute each SubFormula in DAG order
  3. If decision_rule_id set        -> logic_lib.fire(rule, **inputs)
  4. Apply verify_lib sanity bounds  -> block on abstain
  5. Format via format_lib           -> final answer string

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

# ---------------------------------------------------------------------------
# Optional lib imports - all soft
# ---------------------------------------------------------------------------

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

try:
    from . import table_normalizer as _tnorm
    _HAS_TNORM = True
except Exception:
    _tnorm = None
    _HAS_TNORM = False

# Cache the normalized (metric,year)->value map per structured_tables object,
# so we build it ONCE per document rather than per question. Keyed by id() of
# the structured_tables list (stable within one document's lifetime).
_NORM_CACHE: Dict[int, Dict] = {}


def _get_normalized(structured_tables, anchor=None):
    if not _HAS_TNORM or not structured_tables:
        return None
    key = (id(structured_tables), str(anchor))
    cached = _NORM_CACHE.get(key)
    if cached is None:
        try:
            cached = _tnorm.build_normalized(structured_tables,
                                             doc_fiscal_year=anchor)
        except Exception:
            logger.debug("[plan_executor] normalizer failed", exc_info=True)
            cached = {}
        _NORM_CACHE[key] = cached
        if len(_NORM_CACHE) > 16:
            _NORM_CACHE.pop(next(iter(_NORM_CACHE)))
    return cached


# ---------------------------------------------------------------------------
# Number scan
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Raw extraction (cells first, raw_text fallback)
# ---------------------------------------------------------------------------

def _anti_blocks(raw_norm: str, occ, anti_patterns: List[str]) -> bool:
    """Return True if a synonym occurrence should be REJECTED due to an
    anti-pattern.

    FIX-D v2 (2026-06-21): the previous guard only inspected 20 chars to the
    LEFT of the match, but multi-word anti-patterns that overlap the match
    (e.g. non-operating preceding operating income) were never fully
    contained in that left window, so nothing was blocked and operating
    income matched inside non-operating income (-> 13,548 -> 13.5).

    Build a window spanning from up to len(longest_anti) chars BEFORE the
    match through ~24 chars AFTER it; reject if any anti-pattern appears.
    """
    if not anti_patterns:
        return False
    max_anti = max((len(a) for a in anti_patterns), default=0)
    lo = max(0, occ.start() - max_anti)
    hi = min(len(raw_norm), occ.end() + 24)
    window = raw_norm[lo:hi]
    return any(a and a in window for a in anti_patterns)


def _extract_value(
    metric_id: str,
    period:    str,
    cells:     List[Dict],
    raw_text:  str,
    structured_tables: Optional[List[Dict]] = None,
) -> Optional[float]:
    """Try structured tables (column=year) first, then table_cells via
    extract_lib, then fall back to raw_text scan."""
    # Path 0a: NORMALIZED layer (2026-06-21) - root-cause fix. One clean
    # (metric, year)->value map built once per document. Handles year-column
    # detection, unit scaling, and total-vs-segment precedence centrally so
    # we don't re-guess per query.
    if structured_tables and period and _HAS_TNORM:
        norm = _get_normalized(structured_tables, anchor=period)
        if norm:
            v = _tnorm.lookup(norm, metric_id, period)
            if v is not None and _passes_sanity(v, metric_id):
                return v

    # Path 0b: legacy structure-preserving lookup (fallback)
    if structured_tables and period:
        v = _lookup_structured(metric_id, period, structured_tables)
        if v is not None and _passes_sanity(v, metric_id):
            return v

    # Path 1: extract_lib on cells
    if _HAS_EXTRACT and cells:
        try:
            r = _resolve_metric(metric_id, cells, period or "")
            if r and r.valid and r.value is not None:
                v = float(r.value)
                if _passes_sanity(v, metric_id):
                    return v
        except Exception:
            logger.debug("[plan_executor] extract_lib failed", exc_info=True)

    # Path 2: raw_text scan via synonyms
    if not raw_text:
        return None
    syn = _EXT_SYNS.get(metric_id, {}) if _HAS_EXTRACT else {}
    positives = syn.get("positive", [])
    if not positives:
        return None
    # FIX-D part 2 (2026-06-21): apply anti-patterns in the raw_text scan.
    anti_patterns = [_norm(a) for a in
                     (syn.get("anti") or syn.get("negative") or []) if a]

    raw_norm = _norm(raw_text)

    _ym = re.search(r"(19|20)\d{2}", period or "")
    year_tok = _ym.group(0) if _ym else ""

    def _pick_from_window(window: str) -> Optional[float]:
        candidates = []
        for m in re.finditer(_NUMBER_RE, window):
            v = _parse_number(m.group(0))
            if v is None:
                continue
            if _looks_like_year(v):
                continue
            if not _passes_sanity(v, metric_id):
                continue
            candidates.append(v)
        if not candidates:
            return None
        if metric_id in _MEGA_METRICS:
            return max(candidates, key=lambda x: abs(x))
        return candidates[0]

    sorted_syns = sorted(positives, key=len, reverse=True)

    # Pass 1: year-anchored (only when a year was requested)
    if year_tok:
        for synonym in sorted_syns:
            s = _norm(synonym)
            seen = 0
            for occ in re.finditer(re.escape(s), raw_norm):
                seen += 1
                if seen > 50:
                    break
                ctx_start = max(0, occ.start() - 120)
                win_end   = occ.end() + 300
                context   = raw_norm[ctx_start:win_end]
                if year_tok not in context:
                    continue
                if _anti_blocks(raw_norm, occ, anti_patterns):
                    continue
                v = _pick_from_window(raw_norm[occ.end(): win_end])
                if v is not None:
                    return v

    # Pass 2: period-blind first-occurrence behaviour
    for synonym in sorted_syns:
        s = _norm(synonym)
        for occ in re.finditer(re.escape(s), raw_norm):
            if _anti_blocks(raw_norm, occ, anti_patterns):
                continue
            v = _pick_from_window(raw_norm[occ.end(): occ.end() + 300])
            if v is not None:
                return v
    return None


_MEGA_METRICS = {
    "revenue", "total_assets", "total_liabilities",
    "shareholders_equity", "ppe", "cogs",
    "current_assets", "current_liabilities",
    "long_term_debt", "goodwill",
}
_MID_METRICS = {
    "capex", "net_income", "operating_income",
    "gross_profit", "free_cash_flow", "operating_cash_flow",
    "investing_cash_flow", "financing_cash_flow",
    "ebitda", "cash", "inventory", "intangible_assets",
    "income_before_tax", "sg_and_a", "r_and_d",
}
_SMALL_METRICS = {
    "accounts_receivable", "accounts_payable",
    "dividends_paid", "share_repurchases",
    "depreciation_amortization", "interest_expense",
    "income_tax",
}


def _ordered_doc_years(structured_tables: List[Dict]) -> List[str]:
    """Return the document's fiscal years, NEWEST first, as 4-digit strings.

    FIX-E v2 (2026-06-21): the v1 version returned garbage like
    ['2022','2027','2047'] because it accepted ANY run of 4-digit numbers
    (note references, addresses, random figures). A real fiscal-year header
    is a run of CONSECUTIVE years descending by exactly 1 (2019 2018 2017).
    We scan every table's header+rows for the longest such consecutive
    descending run within a plausible range, and return it newest-first.
    """
    from collections import Counter

    def _consec_runs(year_ints: List[int]) -> List[List[int]]:
        """Split a sequence into maximal runs that descend by exactly 1."""
        runs, cur = [], []
        for y in year_ints:
            if cur and cur[-1] - y == 1:
                cur.append(y)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = [y]
        if len(cur) >= 2:
            runs.append(cur)
        return runs

    candidates = Counter()
    for tbl in structured_tables or []:
        cells = list(tbl.get("headers", []) or [])
        for r in (tbl.get("rows", []) or []):
            cells.extend(r)
        ys = []
        for c in cells:
            cs = str(c).strip()
            if re.fullmatch(r"(19|20)\d{2}", cs):
                yi = int(cs)
                if 1995 <= yi <= 2035:        # plausible fiscal range
                    ys.append(yi)
        # dedupe consecutive duplicates while preserving order
        seq = []
        for y in ys:
            if not seq or seq[-1] != y:
                seq.append(y)
        # only descending runs (newest-first as printed in 10-Ks)
        for run in _consec_runs(seq):
            candidates[tuple(run)] += 1
        # also try the reverse (ascending tables)
        for run in _consec_runs(list(reversed(seq))):
            candidates[tuple(run)] += 1

    if not candidates:
        return []
    # longest run wins; ties broken by frequency
    best = max(candidates.items(), key=lambda kv: (len(kv[0]), kv[1]))[0]
    nums = list(best)
    if nums == sorted(nums):              # ascending -> flip to newest-first
        nums = list(reversed(nums))
    return [str(n) for n in nums]


def _positional_row_lookup(
    metric_id: str,
    year: str,
    structured_tables: List[Dict],
    positives: List[str],
    negatives: List[str],
    doc_years: List[str],
) -> Optional[float]:
    """FIX-E (2026-06-21): map a metric row's numbers to years BY POSITION.

    For a row like:
        Consolidated net revenues  $ 6,489  $ 7,500  $ 7,017
    there is no year column header. We pull the numbers in order
    [6489, 7500, 7017] and align them to doc_years [2019, 2018, 2017],
    newest-first, then return the value for the requested year. This lets
    YoY growth see two DIFFERENT period values instead of collapsing to the
    same number (Activision rev 2019 vs 2018).
    """
    if not doc_years or year not in doc_years:
        return None
    pos = doc_years.index(year)
    n_years = len(doc_years)

    # FIX-E v2: collect ALL matching rows, then prefer the best one:
    #   - rows whose label starts with 'total' or 'consolidated' (the
    #     statement total) beat partial/segment rows
    #   - among those, the row whose leading numeric run length best matches
    #     the number of fiscal years wins
    matches = []  # (priority, numbers, label)
    for tbl in structured_tables or []:
        rows = list(tbl.get("rows", []) or [])
        hdr = tbl.get("headers", []) or []
        if hdr:
            rows = [hdr] + rows
        for row in rows:
            if not row:
                continue
            label = _norm(str(row[0]))
            if not label:
                continue
            if any(neg and neg in label for neg in negatives):
                continue
            if not any(pos_s and pos_s in label for pos_s in positives):
                continue
            # collect the LEADING run of numbers (stop at first negative -
            # negatives are change / %-change columns, not a fiscal year value)
            nums = []
            for c in row[1:]:
                v = _parse_number(str(c))
                if v is None or _looks_like_year(v):
                    continue
                if v < 0:
                    break  # change column begins; stop the year run
                nums.append(v)
            if not nums:
                continue
            # priority: total/consolidated rows first
            is_total = label.startswith("total") or label.startswith("consolidated")
            # closeness of run length to number of years (prefer exact)
            close = -abs(len(nums) - n_years)
            priority = (1 if is_total else 0, close)
            matches.append((priority, nums, label))

    if not matches:
        return None
    # best match = highest priority
    matches.sort(key=lambda m: m[0], reverse=True)
    best_nums = matches[0][1]
    if len(best_nums) > pos:
        cand = best_nums[pos]
        if _passes_sanity(cand, metric_id):
            return cand
    return None


def _lookup_structured(
    metric_id: str,
    period:    str,
    structured_tables: List[Dict],
) -> Optional[float]:
    """Structure-preserving (metric, year) lookup (2026-06-20).

    Find the column whose HEADER (or a year-bearing data row) contains the
    requested year, then the row matching a metric synonym, and return that
    cell. Falls back to positional mapping (FIX-E) when no year row exists.
    Deterministic; returns None on any doubt.
    """
    ym = re.search(r"(19|20)\d{2}", period or "")
    if not ym:
        return None
    year = ym.group(0)

    syns = _EXT_SYNS.get(metric_id, {}) if _HAS_EXTRACT else {}
    positives = [_norm(s) for s in syns.get("positive", []) if s]
    # FIX-D (2026-06-21): synonyms file stores anti-patterns under "anti",
    # but this code used to read "negative" -> all anti-patterns were dead
    # code. Read "anti" first, fall back to "negative".
    negatives = [_norm(s) for s in
                 (syns.get("anti") or syns.get("negative") or []) if s]
    if not positives:
        return None

    best: Optional[float] = None
    for tbl in structured_tables or []:
        headers = tbl.get("headers", []) or []
        rows = tbl.get("rows", []) or []
        if not headers or not rows:
            continue

        # column index whose header contains the requested year
        col_idx = None
        for i, h in enumerate(headers):
            if year in str(h):
                col_idx = i
                break

        # FIX-A (2026-06-21): on real 10-Ks the year is usually NOT in the
        # header row; it sits in a DATA row like ['', '2015', '2014']. When
        # the header has no year column, scan data rows for a year-label row
        # and use it as the effective header.
        data_rows = rows
        if col_idx is None:
            year_row_idx = None
            for ri, row in enumerate(rows):
                if not row:
                    continue
                yr_cells = sum(
                    1 for c in row
                    if re.fullmatch(r"(19|20)\d{2}", str(c).strip())
                )
                if yr_cells >= 1:
                    for ci, c in enumerate(row):
                        if year in str(c):
                            col_idx = ci
                            year_row_idx = ri
                            break
                if col_idx is not None:
                    break
            if col_idx is None:
                continue
            data_rows = rows[year_row_idx + 1:] if year_row_idx is not None else rows

        # row whose label matches a metric synonym (and no negative)
        for row in data_rows:
            if not row:
                continue
            label = _norm(str(row[0]))
            if not label:
                continue
            if any(neg and neg in label for neg in negatives):
                continue
            if not any(p and p in label for p in positives):
                continue
            if col_idx < len(row):
                v = _parse_number(str(row[col_idx]))
                if v is not None and not _looks_like_year(v):
                    if best is None or (
                        metric_id in _MEGA_METRICS and abs(v) > abs(best)
                    ):
                        best = v

    # FIX-E: positional fallback when the year-column path found nothing.
    if best is None:
        doc_years = _ordered_doc_years(structured_tables)
        best = _positional_row_lookup(
            metric_id, year, structured_tables, positives, negatives, doc_years
        )

    return best


def _looks_like_year(v: float) -> bool:
    """4-digit integers in 1990-2030 are probably years, not values."""
    if v is None:
        return False
    av = abs(v)
    if 1990 <= av <= 2030 and av == int(av):
        return True
    return False


def _passes_sanity(v: float, metric_id: str) -> bool:
    """Magnitude sanity check with tiered floors per metric type."""
    if v is None:
        return False
    av = abs(v)
    if metric_id in _MEGA_METRICS:
        return av >= 50.0
    if metric_id in _MID_METRICS:
        return av >= 5.0
    if metric_id in _SMALL_METRICS:
        return av >= 1.0
    return True


# ---------------------------------------------------------------------------
# Sub-formula execution
# ---------------------------------------------------------------------------

def _execute_sub(
    sub:           SubFormula,
    extracted:     Dict[str, float],
    intermediate:  Dict[str, float],
    multiplier:    float = 1.0,
    period_values: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> Optional[float]:
    """Compute one SubFormula. Returns the value or None on failure."""
    fid = sub.formula_id
    period_values = period_values or {}

    # MOVE-3: for DIFF / GROWTH on a single metric across two periods, pull
    # the two period-keyed values directly (newest - oldest).
    if sub.operation in (Operation.DIFF, Operation.GROWTH_YOY):
        for metric_id, pvs in period_values.items():
            if len(pvs) >= 2:
                ordered = sorted(pvs, key=lambda pv: pv[0])
                oldest_val = ordered[0][1]
                newest_val = ordered[-1][1]
                if abs(newest_val - oldest_val) < 1e-9:
                    return None
                if sub.operation == Operation.DIFF:
                    return newest_val - oldest_val
                if abs(oldest_val) > 1e-9:
                    return (newest_val - oldest_val) / abs(oldest_val) * 100.0

    # Resolve inputs (from extracted OR intermediate)
    inputs = []
    for name in sub.inputs:
        if name in extracted:
            inputs.append(extracted[name])
        elif name in intermediate:
            inputs.append(intermediate[name])
        else:
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

    for dep in sub.depends_on:
        if dep not in intermediate:
            logger.debug("[plan_executor] unresolved dependency %s for %s", dep, fid)
            return None

    op = sub.operation

    # FIX-C v2 (2026-06-21): per-operand absolute-magnitude scale
    # normalization. A single ratio operand >= 1,000,000 (which would be
    # $1 trillion in millions) must be in thousands -> divide by 1000.
    # Judges each operand alone; never touches plausible million-scale
    # numbers. Fixes Adobe current_liabilities 2,213,556 -> 2,213.556.
    _MILLIONS_CEILING = 1_000_000.0
    if op in (Operation.RATIO, Operation.RATIO_PCT) and len(inputs) >= 2:
        _normed = []
        for _v in inputs:
            if _v is not None and abs(_v) >= _MILLIONS_CEILING:
                _normed.append(_v / 1000.0)
                logger.debug(
                    "[plan_executor] FIX-C v2 normalized thousands->millions: "
                    "%.1f -> %.1f", _v, _v / 1000.0,
                )
            else:
                _normed.append(_v)
        inputs = _normed

    try:
        if op == Operation.RATIO_PCT and len(inputs) >= 2:
            # FIX-B: abstain on a ~0 operand instead of a confident 0.00.
            if abs(inputs[1]) < 1e-9 or abs(inputs[0]) < 1e-9:
                return None
            return abs(inputs[0]) / abs(inputs[1]) * 100.0
        if op == Operation.RATIO and len(inputs) >= 2:
            if abs(inputs[1]) < 1e-9 or abs(inputs[0]) < 1e-9:
                return None
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
            if abs(inputs[0] - inputs[1]) < 1e-9:
                return None
            return (inputs[0] - inputs[1]) / abs(inputs[1]) * 100.0
        if op == Operation.CAGR and len(inputs) >= 2:
            n = max(1, (sub.notes.count("year") or 1))
            return (inputs[0] / abs(inputs[1])) ** (1 / n) - 1
        if op == Operation.PROJECT:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_plan(
    plan:     QuestionPlan,
    cells:    List[Dict],
    raw_text: str,
    company:  str = "",
    fy:       str = "",
    doc_type: str = "",
    structured_tables: Optional[List[Dict]] = None,
) -> ExecutionResult:
    """Execute the plan deterministically using all available libs. Never raises."""
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

    extracted: Dict[str, float] = {}
    period_values: Dict[str, List[Tuple[str, float]]] = {}
    for req in plan.required_extracts:
        val = _extract_value(req.metric_id, req.period, cells, raw_text,
                             structured_tables=structured_tables)
        if val is not None:
            key = f"{req.metric_id}_{req.period}" if req.period else req.metric_id
            extracted[key] = val
            extracted.setdefault(req.metric_id, val)
            period_values.setdefault(req.metric_id, []).append(
                (req.period or "", val)
            )
    result.audit_trail["extracted"] = dict(extracted)
    result.audit_trail["period_values"] = {
        k: list(v) for k, v in period_values.items()
    }

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
        val = _execute_sub(sub, extracted, intermediate, multiplier=multiplier,
                           period_values=period_values)
        if val is None:
            result.audit_trail.setdefault("failed_subs", []).append(sub.name)
            continue
        intermediate[sub.name] = val
    result.intermediate_values = dict(intermediate)

    final_value: Optional[float] = None
    if plan.sub_formulas:
        last = sorted_subs[-1] if sorted_subs else None
        if last and last.name in intermediate:
            final_value = intermediate[last.name]
    elif extracted:
        if plan.subject and plan.subject.metric_id in extracted:
            final_value = extracted[plan.subject.metric_id]

    if final_value is None:
        result.audit_trail["reason"] = "no_final_value"
        return result

    classification = ""
    if plan.decision_rule_id and _HAS_LOGIC and _logic_fire is not None:
        try:
            rule_input_name = _guess_rule_input_name(plan.decision_rule_id)
            r = _logic_fire(plan.decision_rule_id, **{rule_input_name: final_value})
            if r and getattr(r, "fired", False):
                classification = str(getattr(r, "output", ""))
                result.audit_trail["decision_branch"] = getattr(r, "branch", "")
        except Exception:
            logger.exception("[plan_executor] logic_lib.fire failed")

    if (
        _HAS_VERIFY and _verify is not None
        and plan.intent.value != "extract"
        and plan.sub_formulas
    ):
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

    unit = plan.output_unit or ""
    if classification:
        cap_class = classification[:1].upper() + classification[1:]
        unit_suffix = "%" if "%" in unit or "_pct" in (plan.operation.value or "") else ""
        result.final_answer = (
            f"{cap_class}. {plan.subject.display_name if plan.subject else plan.output_metric}: "
            f"{final_value:.2f}{unit_suffix}."
        )
    else:
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
    value = abs(float(value))

    if unit == "$B":
        scaled = value / 1000.0
        return f"${scaled:,.2f} billion"
    if unit == "$K":
        scaled = value * 1000.0
        return f"${scaled:,.0f} thousand"

    if _HAS_FORMAT and _format_render is not None:
        try:
            r = _format_render("default_number", value)
            if r is not None:
                txt = getattr(r, "text", None) or getattr(r, "value", None) or str(r)
                if unit == "%" and "%" not in str(txt):
                    return f"{txt}%"
                return str(txt)
        except Exception:
            pass
    if unit == "%":
        return f"{value:.2f}%"
    if unit in ("x", ""):
        return f"{value:.2f}"
    if unit in ("$", "$M"):
        return f"${value:,.2f} million" if unit == "$M" else f"${value:.2f}"
    return f"{value:.2f} {unit}"


def _guess_rule_input_name(rule_id: str) -> str:
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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .decomposer import parse_question

    print("plan_executor - self test (offline, libs may be missing)")
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
