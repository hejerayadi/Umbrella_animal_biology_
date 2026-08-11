from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class AgentStatus(Enum):
    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


@dataclass
class AgentRequest:
    instruction: str
    context: dict[str, Any]


@dataclass
class AgentResult:
    status: AgentStatus
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: Any | None = None


@dataclass
class ReconstructionInput(AgentRequest):
    genome_sequence: str
    species_metadata: dict[str, Any]


@dataclass
class ValidatedGenome:
    cleaned_sequence: str
    species_id: str
    species_metadata: dict[str, Any]
    sequence_type: Literal["nuclear", "mitochondrial"]
    validation_notes: str | None = None


@dataclass
class GapRegion:
    start: int
    end: int
    length: int
    in_scope: bool
    sequence_type: Literal["nuclear", "mitochondrial"] = "nuclear"  # inherited from ValidatedGenome by Gap Locator


@dataclass
class GapPrediction:
    gap: GapRegion
    predicted_sequence: str
    model_used: str
    confidence: float


@dataclass
class ValidationResult:
    similarity_score: float | None = None
    related_species_used: str | None = None
    plausibility_flag: bool = True


@dataclass
class ReconstructedGenome:
    species_id: str
    reconstructed_sequence: str
    predictions: list[GapPrediction]
    excluded_gaps: list[GapRegion]
    is_partial: bool


@dataclass
class ReconstructionResult(AgentResult):
    species_id: str | None = None
    reconstructed_sequence: str | None = None
    gaps_found: int | None = None
    gaps_reconstructed: int | None = None
    predictions: list[GapPrediction] | None = None
    excluded_gaps: list[GapRegion] | None = None
    overall_confidence: float | None = None
    is_partial: bool = False
    error_code: str | None = None   # machine-readable error type (e.g. "INPUT_VALIDATION_FAILED", "MAX_RETRIES_EXCEEDED", "EVOLUTION_AGENT_TIMEOUT")
    notes: str | None = None
