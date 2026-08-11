from __future__ import annotations

import logging
import os
import re
import json
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .retrieval_pipeline import RetrievalPipeline
from .data_ingestion.dna_models.router import DNAModelRouter
from .data_ingestion.dna_models.utils import FlanksWindow

# Import the shared contracts
from .schema import (
    AgentRequest, 
    AgentResult, 
    AgentStatus,
    ValidatedGenome,
    GapRegion,
    GapPrediction,
    ValidationResult,
    ReconstructionResult
)

logger = logging.getLogger(__name__)

class ReconstructionState(TypedDict, total=False):
    request: AgentRequest
    validated_genome: ValidatedGenome
    gaps: List[GapRegion]
    predictions: List[GapPrediction]
    excluded_gaps: List[GapRegion]
    current_gap_index: int
    previous_attempts: List[str]  # models attempted for the CURRENT gap (max MAX_ATTEMPTS=3)
    current_prediction: Optional[GapPrediction]
    validation_result: Optional[ValidationResult]  # last ValidationResult produced by validation_engine
    needs_evolution_agent: bool
    is_partial: bool
    assembled_sequence: str
    result: Optional[AgentResult]


def input_manager(state: ReconstructionState) -> ReconstructionState:
    """Validates FASTA, verifies species metadata, sets sequence_type and is_extinct."""
    if "validated_genome" in state:
        return state

    request = state["request"]
    genome = request.context.get("genome", "")
    species_metadata = request.context.get("species_metadata", {})
    
    # Input Manager makes the authoritative decision on is_extinct.
    # Extinction status lives exclusively in species_metadata["is_extinct"] —
    # there is no top-level flat key.  If the key is absent from species_metadata
    # it defaults to False (living species assumption).
    is_extinct = bool(species_metadata.get("is_extinct", False))
    species_metadata["is_extinct"] = is_extinct
    
    sequence_type = request.context.get("sequence_type", "nuclear")

    if not genome or len(genome) < 10:
        state["result"] = AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="Genome",
            prompt_to_target_agent="Retrieve the incomplete genome sequence."
        )
        return state

    state["validated_genome"] = ValidatedGenome(
        cleaned_sequence=genome.replace("\n", "").replace(" ", "").upper(),
        species_id=species_metadata.get("species_id", request.context.get("species", "unknown")),
        species_metadata=species_metadata,
        sequence_type=sequence_type,
        validation_notes="Validated successfully."
    )
    return state


def gap_locator(state: ReconstructionState) -> ReconstructionState:
    """Identifies N-regions and sets in_scope (True only for 10-500bp gaps).
    Propagates sequence_type from ValidatedGenome to every GapRegion so that
    downstream nodes can stratify by nuclear / mitochondrial without re-querying
    the validated genome.
    """
    # Skip only when the key is ABSENT from state — that is the sole reliable
    # sentinel that gap_locator has already run for this thread.  Once it runs
    # it always writes "gaps" (even as [] for a gap-free sequence), so key
    # presence means "already ran".  A pre-seeded gaps=[] or gaps=None must
    # NOT be treated as "already ran" — both are artefacts of test fixtures or
    # stale state seeds and must trigger a real gap-location pass.
    if "gaps" in state and state["gaps"] is not None:
        # Only truly skip if we also have the companion fields that gap_locator
        # writes in the same pass.  If they are absent the state was partially
        # seeded and we must re-run.
        if "excluded_gaps" in state and "current_gap_index" in state:
            return state

    vg = state["validated_genome"]
    seq = vg.cleaned_sequence
    sequence_type = vg.sequence_type  # propagate to all gap regions

    gaps = []
    excluded_gaps = []
    
    for match in re.finditer(r'N+', seq):
        start = match.start()
        end = match.end()
        length = end - start
        in_scope = 10 <= length <= 500
        gap = GapRegion(start=start, end=end, length=length, in_scope=in_scope, sequence_type=sequence_type)
        if in_scope:
            gaps.append(gap)
        else:
            excluded_gaps.append(gap)

    state["gaps"] = gaps
    state["excluded_gaps"] = excluded_gaps
    state["predictions"] = []
    state["current_gap_index"] = 0
    state["previous_attempts"] = []
    state["needs_evolution_agent"] = False

    return state


def parse_llm_prediction(response: str, gap_length: int) -> tuple[str, float]:
    """Parses JSON from LLM response or returns a fallback."""
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            data = json.loads(json_str)
            seq = data.get("predicted_sequence", "N" * gap_length)
            conf = float(data.get("confidence", 0.0))
            return seq, conf
    except Exception as exc:
        logger.warning("Failed to parse LLM prediction for gap length %s: %s", gap_length, exc)

    logger.warning("Falling back to default prediction for gap length %s", gap_length)
    return "N" * gap_length, 0.0


def model_selector_predictor(state: ReconstructionState) -> ReconstructionState:
    """Routes gap prediction through DNAModelRouter (DNABERT → NT → GPT fallback).

    Replaces the hard-coded Azure-GPT-5.1 call with a cascade of DL models.
    Only state["current_prediction"] and state["previous_attempts"] are modified.
    All other state keys and the LangGraph node contract are preserved unchanged.

    Requirements: 1.1–1.7, 6.1–6.6, 10.1–10.4
    """
    idx = state.get("current_gap_index", 0)
    gaps = state.get("gaps", [])
    if idx >= len(gaps):
        # Req 6.3: no gap to predict — return state unchanged
        return state

    gap = gaps[idx]
    vg = state["validated_genome"]
    seq = vg.cleaned_sequence

    flank_size = 1000
    start_flank = seq[max(0, gap.start - flank_size):gap.start]
    end_flank = seq[gap.end:min(len(seq), gap.end + flank_size)]

    species_metadata = vg.species_metadata

    # Req 10.1: try RetrievalPipeline first; fall back to direct sequence slice
    try:
        pipeline = RetrievalPipeline(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "reconstruction_windows"),
            top_k=5,
        )
        left_context, right_context = pipeline.get_flanking_context(
            species_id=vg.species_id,
            gap_start=gap.start,
            gap_end=gap.end,
            cleaned_sequence=seq,
            species_metadata=species_metadata,
        )
        if not left_context:
            left_context = start_flank
        if not right_context:
            right_context = end_flank
    except Exception:
        left_context, right_context = start_flank, end_flank

    # Req 10.2, 10.4: build FlanksWindow with sequence_type propagated
    flanks = FlanksWindow(
        left_context=left_context,
        gap_length=gap.length,
        right_context=right_context,
        sequence_type=gap.sequence_type,
    )

    previous_attempts = list(state.get("previous_attempts", []))

    # Req 10.3: skip DL if total context too short — router handles via AzureGPT fallback
    # (DNAModelRouter._azure_gpt_predict is the fallback for total_length < 20)

    # Req 1.1–1.7, 6.1–6.6: delegate to cascade router
    prediction = DNAModelRouter.get().predict(
        flanks=flanks,
        gap=gap,
        previous_attempts=previous_attempts,
    )

    # Req 6.2: only write current_prediction and previous_attempts
    state["current_prediction"] = prediction
    state["previous_attempts"] = previous_attempts + [prediction.model_used]
    return state


def validation_engine(state: ReconstructionState) -> ReconstructionState:
    """Evaluates the prediction via a ValidationResult, sets needs_evolution_agent
    flag if cross-species delegation is required, and enforces the max_attempts cap
    (3 retries) with excluded-gap fallback on exhaustion.
    """
    prediction = state.get("current_prediction")
    if not prediction:
        return state

    plausibility_flag = prediction.confidence > 0.7
    vg = state["validated_genome"]
    is_extinct = vg.species_metadata.get("is_extinct", False)
    
    evo_analysis = state["request"].context.get("evolution_analysis")
    
    if is_extinct and not evo_analysis:
        # Require delegation to Evolution Agent — do not produce ValidationResult yet.
        state["needs_evolution_agent"] = True
        return state

    # If we have evolution_analysis, it contributes to plausibility
    if is_extinct and evo_analysis:
        evo_str = evo_analysis if isinstance(evo_analysis, str) else str(evo_analysis)
        plausibility_flag = plausibility_flag and bool(evo_str.strip())

    # Build the ValidationResult now that we have enough information.
    # (similarity_score / related_species_used are populated by evolution_agent_delegate;
    #  for living species they remain None.)
    validation_result = ValidationResult(
        plausibility_flag=plausibility_flag,
    )
    state["validation_result"] = validation_result
    state["needs_evolution_agent"] = False

    MAX_ATTEMPTS = 3  # tunable — see description.md Section 5

    if plausibility_flag:
        if "predictions" not in state:
            state["predictions"] = []
        predictions = list(state.get("predictions", []))
        predictions.append(prediction)
        state["predictions"] = predictions
        # Move to next gap and reset retry tracking
        state["current_gap_index"] = state.get("current_gap_index", 0) + 1
        state["previous_attempts"] = []
        state["current_prediction"] = None
    else:
        # Retry logic: clear current prediction so reasoning_core re-routes to
        # model_selector_predictor.  If max_attempts reached, flag the gap as
        # a partial / excluded gap and advance the index.
        state["current_prediction"] = None
        if len(state.get("previous_attempts", [])) >= MAX_ATTEMPTS:
            # Abandon gap — flag as partial assembly (excluded gap upon failure)
            gap = prediction.gap
            excluded = list(state.get("excluded_gaps", []))
            excluded.append(gap)
            state["excluded_gaps"] = excluded
            state["is_partial"] = True
            state["current_gap_index"] = state.get("current_gap_index", 0) + 1
            state["previous_attempts"] = []

    return state


def evolution_agent_delegate(state: ReconstructionState) -> ReconstructionState:
    """Node that triggers the cross-species delegation to the Evolution Agent.
    
    Reached via a conditional edge from validation_engine (not from reasoning_core).
    Times out after the Evolution Agent response window (handled upstream by the
    orchestrator); if the orchestrator receives no response within its timeout,
    it resumes with an empty evolution_analysis and the validation_engine will
    flag the gap as partial/excluded rather than hanging indefinitely.
    """
    state["result"] = AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        target_agent="Evolution",
        prompt_to_target_agent="Compare cette séquence prédite à l'espèce vivante la plus proche et retourne un score de similarité"
    )
    # The flag is cleared so when we resume, we go back to reasoning_core and then validation_engine
    state["needs_evolution_agent"] = False
    return state


def prediction_assembler(state: ReconstructionState) -> ReconstructionState:
    """Reassembles the sequence from resolved GapPredictions."""
    vg = state["validated_genome"]
    seq = vg.cleaned_sequence
    predictions = state.get("predictions", [])
    excluded_gaps = state.get("excluded_gaps", [])
    gaps = state.get("gaps", [])
    
    # Sort backwards to avoid index shifting during replacement
    sorted_preds = sorted(predictions, key=lambda p: p.gap.start, reverse=True)
    
    assembled = seq
    for p in sorted_preds:
        assembled = assembled[:p.gap.start] + p.predicted_sequence + assembled[p.gap.end:]
        
    resolved_starts = {p.gap.start for p in predictions}
    is_partial = len(excluded_gaps) > 0
    
    for g in gaps:
        if g.start not in resolved_starts:
            is_partial = True
            break
            
    state["is_partial"] = is_partial
    state["assembled_sequence"] = assembled
    return state


def output_formatter(state: ReconstructionState) -> ReconstructionState:
    """Returns final ReconstructionResult, carrying is_partial and error_code."""
    vg = state.get("validated_genome")
    gaps = state.get("gaps", [])
    excluded_gaps = state.get("excluded_gaps", [])
    predictions = state.get("predictions", [])
    is_partial = state.get("is_partial", False)
    
    if not vg:
        # Fallback if input_manager failed
        res = ReconstructionResult(
            status=AgentStatus.FAILED,
            error_code="INPUT_VALIDATION_FAILED",
            notes="Input validation failed."
        )
        state["result"] = res
        return state

    overall_conf = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0.0
    
    error_code = None
    notes = "Reconstruction complete."
    if is_partial:
        error_code = "PARTIAL_ASSEMBLY"
        notes = "Partial reconstruction. Some gaps were unresolved or out of scope."
        
    res = ReconstructionResult(
        status=AgentStatus.COMPLETED,
        species_id=vg.species_id,
        reconstructed_sequence=state.get("assembled_sequence", vg.cleaned_sequence),
        gaps_found=len(gaps) + len(excluded_gaps),
        gaps_reconstructed=len(predictions),
        predictions=predictions,
        excluded_gaps=excluded_gaps,
        overall_confidence=overall_conf,
        is_partial=is_partial,
        error_code=error_code,
        notes=notes
    )
    
    state["result"] = res
    return state


def reasoning_core(state: ReconstructionState) -> str:
    """
    The central hub of the LangGraph StateGraph.
    Inspects state and dynamically routes to the next node.
    """
    result = state.get("result")

    # If a NEEDS_AGENT result is present, check whether this is a resumption.
    # On resumption the orchestrator injects `evolution_analysis` into the
    # request context.  In that case the stale NEEDS_AGENT result must be
    # cleared so the graph continues instead of halting immediately.
    if result is not None and getattr(result, "status", None) == AgentStatus.NEEDS_AGENT:
        evo_analysis = state.get("request") and state["request"].context.get("evolution_analysis")
        if not evo_analysis:
            # Still waiting — halt so the orchestrator can route to Evolution Agent.
            return END
        # Resumption: clear stale result and continue
        state["result"] = None
        result = None

    # 1. Ensure input is validated
    if "validated_genome" not in state:
        return "input_manager"
        
    # 2. Locate gaps — route to gap_locator only when the key is absent entirely
    # OR when it was seeded without the companion fields (partially initialised
    # state).  Once gap_locator has run it always writes "gaps", "excluded_gaps",
    # and "current_gap_index" in the same pass, so all three being present is
    # the reliable "already ran" sentinel.
    if not ("gaps" in state and state["gaps"] is not None
            and "excluded_gaps" in state and "current_gap_index" in state):
        return "gap_locator"
        
    gaps = state.get("gaps", [])
    current_idx = state.get("current_gap_index", 0)
    
    # 3. Process gaps
    if current_idx < len(gaps):
        # We are working on a gap
        if state.get("current_prediction") is None:
            # Need to predict
            return "model_selector_predictor"
        else:
            # Have a prediction, need to validate
            return "validation_engine"

    # 4. If all gaps processed but not assembled
    if "assembled_sequence" not in state:
        return "prediction_assembler"
        
    # 5. Finally, format output
    if result is None or getattr(result, "status", None) != AgentStatus.COMPLETED:
        return "output_formatter"

    return END


def check_evolution_needs(state: ReconstructionState) -> str:
    """Conditional edge out of validation_engine."""
    if state.get("needs_evolution_agent"):
        return "evolution_agent_delegate"
    return "reasoning_core"


def reasoning_core_node(state: ReconstructionState) -> ReconstructionState:
    """Dummy node representing the reasoning core. The actual logic is in the edge router."""
    return state


def build_graph():
    graph = StateGraph(ReconstructionState)
    
    graph.add_node("input_manager", input_manager)
    graph.add_node("gap_locator", gap_locator)
    graph.add_node("model_selector_predictor", model_selector_predictor)
    graph.add_node("validation_engine", validation_engine)
    graph.add_node("evolution_agent_delegate", evolution_agent_delegate)
    graph.add_node("prediction_assembler", prediction_assembler)
    graph.add_node("output_formatter", output_formatter)
    
    # Central hub node
    graph.add_node("reasoning_core", reasoning_core_node)
    
    # Entry goes to hub
    graph.set_entry_point("reasoning_core")
    
    # The hub routes to all other nodes
    graph.add_conditional_edges(
        "reasoning_core",
        reasoning_core,
        {
            "input_manager": "input_manager",
            "gap_locator": "gap_locator",
            "model_selector_predictor": "model_selector_predictor",
            "validation_engine": "validation_engine",
            "prediction_assembler": "prediction_assembler",
            "output_formatter": "output_formatter",
            END: END
        }
    )
    
    # All standard tools route back to the hub, except validation_engine which checks for delegation
    graph.add_edge("input_manager", "reasoning_core")
    graph.add_edge("gap_locator", "reasoning_core")
    graph.add_edge("model_selector_predictor", "reasoning_core")
    
    graph.add_conditional_edges(
        "validation_engine",
        check_evolution_needs,
        {
            "evolution_agent_delegate": "evolution_agent_delegate",
            "reasoning_core": "reasoning_core"
        }
    )
    
    # After delegation, return to hub
    graph.add_edge("evolution_agent_delegate", "reasoning_core")
    
    graph.add_edge("prediction_assembler", "reasoning_core")
    graph.add_edge("output_formatter", "reasoning_core")
    
    return graph

# Export compiled graph at module level with checkpointer
graph = build_graph().compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    # Test script adapted to use graph directly (without persistent state files)
    print("Run via api.py for persistent state execution.")
