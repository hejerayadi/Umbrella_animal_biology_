"""Intent classification — backward-compatible re-export.

The real implementation is in planner.py.
This module re-exports for backward compatibility with existing imports.
"""

from __future__ import annotations

from .planner import (
    _extract_json,
    _clean_str,
    _clean_list,
    _apply_guards,
    plan,
    PLANNER_SYSTEM_PROMPT,
)
from .schema import PlannerDecision, PlannedFeature

# Backward-compatible aliases
RecognizedIntent = PlannerDecision
classify_intent = plan
_to_intent = _apply_guards
INTENT_SYSTEM_PROMPT = PLANNER_SYSTEM_PROMPT

__all__ = [
    "classify_intent",
    "plan",
    "RecognizedIntent",
    "PlannerDecision",
    "PlannedFeature",
    "_extract_json",
    "_clean_str",
    "_clean_list",
    "_to_intent",
    "INTENT_SYSTEM_PROMPT",
]
