"""
question_lib/models.py
All dataclasses + enums shared across the question-understanding pipeline.

100% stdlib. No external deps. Used by every other module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """Top-level intent: what does the user want?"""
    EXTRACT   = "extract"      # "What was Apple's revenue in FY2023?"
    COMPUTE   = "compute"      # "What was Apple's gross margin in FY2023?"
    DECIDE    = "decide"       # "Is 3M capital-intensive?"
    COMPARE   = "compare"      # "Compare Apple vs MSFT revenue"
    NARRATE   = "narrate"      # "What drove margin change?"
    PROJECT   = "project"      # "If growth stays at 5%, what's revenue in FY2024?"
    UNKNOWN   = "unknown"


class Operation(str, Enum):
    """Operation type within the intent."""
    LOOKUP     = "lookup"       # direct extraction
    RATIO      = "ratio"        # A / B
    RATIO_PCT  = "ratio_pct"    # A / B * 100
    DIFF       = "diff"         # A - B
    SUM        = "sum"          # A + B
    PRODUCT    = "product"      # A * B
    GROWTH_YOY = "growth_yoy"   # (A - B) / B
    CAGR       = "cagr"
    CLASSIFY   = "classify"     # apply decision rule
    EXPLAIN    = "explain"      # narrative extraction
    PROJECT    = "project"      # extrapolation
    UNKNOWN    = "unknown"


class Polarity(str, Enum):
    """Direction of the question (for decision questions)."""
    POSITIVE = "positive"   # "is X good/healthy/strong"
    NEGATIVE = "negative"   # "is X bad/weak/declining"
    NEUTRAL  = "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Subject:
    """Main metric subject of the question."""
    metric_id: str                          # extract_lib canonical id (e.g. "revenue")
    display_name: str                       # human label (e.g. "gross margin")
    matched_phrase: str = ""                # the actual phrase from question
    char_start: int = 0
    char_end: int = 0
    confidence: float = 1.0


@dataclass
class Period:
    """Time anchor extracted from a question."""
    raw: str                                # original substring (e.g. "FY2022")
    fiscal_year: Optional[str] = None       # canonicalised (e.g. "2022")
    quarter: Optional[str] = None           # "Q1" / "Q2" / "Q3" / "Q4"
    calendar_year: Optional[int] = None
    is_year_end: bool = False               # "year-end FY2022"
    is_ttm: bool = False                    # trailing twelve months
    is_range: bool = False                  # "between FY22 and FY23"
    range_end: Optional["Period"] = None
    confidence: float = 1.0


@dataclass
class Modifier:
    """Side-words: excluding, assuming, 2x, between, per, same-as, ..."""
    kind: str                                # see ModifierKind constants
    raw: str                                 # original phrase
    value: Optional[float] = None            # numeric value (e.g. 2.0 for "2x")
    target: str = ""                         # what the modifier targets
    polarity: Polarity = Polarity.NEUTRAL
    confidence: float = 1.0


class ModifierKind:
    """String constants for Modifier.kind (avoid enum overhead for hot path)."""
    EXCLUDING        = "excluding"           # "excluding M&A"
    ASSUMING         = "assuming"            # "assuming gross margin..."
    CONDITION_IF     = "condition_if"        # "if revenue grows at..."
    MULTIPLIER       = "multiplier"          # "2x the rate", "twice as fast"
    BETWEEN          = "between"             # "between FY22 and FY23"
    PER              = "per"                 # "per share", "per employee"
    SAME_AS          = "same_as"             # "at the same rate as"
    COMPARED_TO      = "compared_to"         # "compared to last year"
    CHANGE_IN        = "change_in"           # "change in margin"
    NEGATION         = "negation"            # "not", "without"
    AVERAGE_OF       = "average_of"          # "average of last 3 years"
    EXCLUDING_MA     = "excluding_ma"        # specifically "excluding M&A"
    ORGANIC          = "organic"             # "organic growth"


@dataclass
class SubFormula:
    """One sub-formula in a multi-step plan (the JEE banana-peel)."""
    name: str                                # short id e.g. "revenue_growth_yoy"
    formula_id: str                          # matched maths_lib formula id
    inputs: List[str] = field(default_factory=list)        # extract metric ids
    depends_on: List[str] = field(default_factory=list)    # other SubFormula.name's
    operation: Operation = Operation.UNKNOWN
    period: Optional[Period] = None
    modifiers: List[Modifier] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExtractRequest:
    """One raw value the executor needs to fetch."""
    metric_id: str                           # extract_lib id
    period: str = ""                         # fiscal year hint
    segment: str = ""                        # optional segment filter
    label: str = ""                          # readable label for audit


@dataclass
class QuestionPlan:
    """The complete parsed-and-decomposed question.

    This is what `parse_question(q)` returns. `execute_plan(plan)` then
    uses extract_lib + maths_lib + logic_lib + format_lib to compute the
    answer deterministically (no LLM).
    """
    raw_question: str
    intent: Intent = Intent.UNKNOWN
    operation: Operation = Operation.UNKNOWN
    polarity: Polarity = Polarity.NEUTRAL
    subject: Optional[Subject] = None
    periods: List[Period] = field(default_factory=list)
    modifiers: List[Modifier] = field(default_factory=list)
    sub_formulas: List[SubFormula] = field(default_factory=list)
    required_extracts: List[ExtractRequest] = field(default_factory=list)
    output_metric: str = ""
    output_unit: str = ""                    # "$M", "%", "x", "days", etc.
    decision_rule_id: str = ""               # logic_lib L02 rule if intent=DECIDE
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    audit_trail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Output of plan_executor.execute_plan(plan)."""
    answered: bool
    final_answer: str = ""
    final_value: Optional[float] = None
    final_unit: str = ""
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    used_libs: List[str] = field(default_factory=list)
    audit_trail: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity: ensure every dataclass instantiates with defaults
    print("question_lib.models — self test")
    plan = QuestionPlan(raw_question="Is 3M capital-intensive?")
    plan.intent = Intent.DECIDE
    plan.subject = Subject(metric_id="capex", display_name="capex")
    plan.periods.append(Period(raw="FY2022", fiscal_year="2022"))
    plan.modifiers.append(
        Modifier(kind=ModifierKind.MULTIPLIER, raw="2x", value=2.0)
    )
    plan.sub_formulas.append(
        SubFormula(name="capital_intensity", formula_id="capex_to_revenue",
                   operation=Operation.RATIO_PCT)
    )
    plan.required_extracts.append(
        ExtractRequest(metric_id="capex", period="2022")
    )
    print(f"  intent      = {plan.intent}")
    print(f"  subject     = {plan.subject.display_name}")
    print(f"  modifiers   = {[m.kind for m in plan.modifiers]}")
    print(f"  sub_formulas= {[s.name for s in plan.sub_formulas]}")
    print("  models.py OK")
