from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

logger = logging.getLogger(__name__)


@dataclass
class GenomeAgentState:
    user_question: str = ""
    needs_metadata: bool = False
    needs_annotation: bool = False
    species_name: str = ""
    visualization_scope: str = "chromosome_map"
    species: dict | None = None
    assembly_id: str | None = None
    metadata: dict | None = None
    annotation: dict | None = None
    visualization: dict | None = None
    explanation: str | None = None
    errors: Annotated[list[str], operator.add] = field(default_factory=list)
    waiting_stack: list[str] = field(default_factory=list)
    waiting_agent: str | None = None
    _metadata_done: bool = False
    _annotation_done: bool = False
