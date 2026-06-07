"""
fina_question_lib — Deterministic question-understanding for financial QA.

The JEE side-word method, made into a library:
  1. Detect intent  (EXTRACT / COMPUTE / DECIDE / COMPARE / NARRATE / PROJECT)
  2. Extract subject (revenue, margin, capex, ...)
  3. Extract periods (FY22, Q4, between FY21 and FY23)
  4. Extract modifiers / side-words (excluding, assuming, 2x, per share, ...)
  5. Detect operation (LOOKUP / RATIO / DIFF / PROJECT / ...)
  6. Match a maths_lib formula signature
  7. Decompose multi-step → sub-formula DAG
  8. Execute deterministically via extract_lib + maths_lib + logic_lib + format_lib + verify_lib

100% deterministic. NO LLM. < 100 ms for typical questions.

Quick start:
    from question_lib import answer_question
    result = answer_question(
        "Is 3M a capital-intensive business in FY2022?",
        raw_text=document_text,
        company="3M",
        fiscal_year="FY2022",
        doc_type="10-K",
    )
    print(result.answered, result.final_answer)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .registry import (
    answer_question,
    describe_plan,
    execute_plan,
    lib_status,
    parse_question,
)
from .models import (
    ExecutionResult,
    ExtractRequest,
    Intent,
    Modifier,
    ModifierKind,
    Operation,
    Period,
    Polarity,
    QuestionPlan,
    Subject,
    SubFormula,
)

__all__ = [
    # API
    "answer_question",
    "describe_plan",
    "execute_plan",
    "lib_status",
    "parse_question",
    # Models
    "ExecutionResult",
    "ExtractRequest",
    "Intent",
    "Modifier",
    "ModifierKind",
    "Operation",
    "Period",
    "Polarity",
    "QuestionPlan",
    "Subject",
    "SubFormula",
]
