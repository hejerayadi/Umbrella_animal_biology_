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
    visualization_scope: str = ""
    species: dict | None = None
    assembly_id: str | None = None
    metadata: dict | None = None
    annotation: dict | None = None
    visualization: dict | None = None
    reconstruction_need: dict | None = None   # ← NEW
    sequence_accession: str | None = None     # ← NEW: real NW_.../NC_... accession, resolved by find_target_gaps_node
    target_gaps: list[dict] | None = None     # ← NEW: [{start, end, length, left_flank, right_flank}, ...]
    explanation: str | None = None
    errors: Annotated[list[str], operator.add] = field(default_factory=list)
    waiting_stack: list[str] = field(default_factory=list)
    waiting_agent: str | None = None
    _metadata_done: bool = False
    _annotation_done: bool = False

    # ── Evaluation instrumentation (Sprint 4, Part A Step 4) ──────────────
    # Every node appends its own name here on every return path (see the
    # _track() wrapper in orchestrator.py) so run_eval.py can reconstruct
    # the actual trajectory without hooking LangGraph's event stream.
    # operator.add is correct here (not overwrite) because parallel branches
    # (get_genome_metadata / get_gene_annotation) each contribute their own
    # entry and LangGraph must concatenate, not clobber, the two.
    node_sequence: Annotated[list[str], operator.add] = field(default_factory=list)
    # Tool-call boundary log: [{"tool": "ncbi_taxonomy_search", "args": {...}}, ...]
    # Populated by the node that owns the call (not the subagent itself,
    # which has no access to graph state) — see species_resolver_node.py and
    # genome_data_nodes.py. Logged *before* the underlying call so a call
    # that times out or raises still shows up as "attempted" for
    # check_tool_selection / check_efficiency.
    tool_calls_log: Annotated[list[dict], operator.add] = field(default_factory=list)