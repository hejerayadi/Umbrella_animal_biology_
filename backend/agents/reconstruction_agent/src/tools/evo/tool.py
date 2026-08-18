"""Evolutionary context and Evo 2 plausibility scoring.

Two tools with different jobs:

- `EvolutionaryContextTool` ranks reference organisms by relatedness. Which
  relatives are informative is a phylogenetics question, and the Umbrella
  system already has an agent for it.
- `Evo2PlausibilityTool` asks a genomic foundation model whether a proposed
  reconstruction reads as real sequence - evidence homology cannot provide.
"""
from __future__ import annotations

from configuration.logging import get_logger
from domain.exceptions import ReconstructionError
from infrastructure.nvidia.client import Continuation, Evo2Client
from tools.contracts import Tool
from tools.evo.mapper import heuristic_relatedness
from tools.evo.schemas import (
    CandidateAgreement,
    EvolutionaryContextInput,
    EvolutionaryContextOutput,
    PlausibilityInput,
    PlausibilityOutput,
)

_log = get_logger(__name__)

# Below this, the local heuristic is admitting it cannot place the organism
# rather than asserting distance - the point at which delegating pays off.
_DELEGATION_THRESHOLD = 0.5


class EvolutionaryContextTool(Tool[EvolutionaryContextInput, EvolutionaryContextOutput]):
    """Scores how closely candidate organisms relate to the target.

    Runs a name-based heuristic locally. When that heuristic cannot separate
    the candidates, it says so in `notes`, and the graph may return
    NEEDS_AGENT to have the orchestrator route the question to the Evolution
    Agent - which is what `card.json` declares this agent may need.
    """

    name = "evolutionary_context"
    description = (
        "Score how closely candidate reference organisms relate to the target organism, so "
        "that references can be ranked by phylogenetic proximity as well as by alignment "
        "quality. Fast and local, but genus-level only; when it cannot separate candidates "
        "the Evolution Agent should be consulted instead."
    )
    estimated_seconds = 0.1

    async def run(self, payload: EvolutionaryContextInput) -> EvolutionaryContextOutput:
        if not payload.target_organism:
            return EvolutionaryContextOutput(
                succeeded=True,
                notes=["No target organism given; relatedness could not be estimated."],
            )

        relatedness = {
            organism: heuristic_relatedness(payload.target_organism, organism)
            for organism in payload.candidate_organisms
        }

        notes: list[str] = []
        if relatedness and max(relatedness.values()) < _DELEGATION_THRESHOLD:
            notes.append(
                "The name-based heuristic could not place any candidate near "
                f"{payload.target_organism}. A phylogeny from the Evolution Agent would "
                "rank these references far better."
            )

        return EvolutionaryContextOutput(
            succeeded=True,
            relatedness=relatedness,
            source="heuristic",
            notes=notes,
            diagnostics={"delegation_recommended": bool(notes)},
        )


class Evo2PlausibilityTool(Tool[PlausibilityInput, PlausibilityOutput]):
    """Checks candidate reconstructions against what Evo 2 expects in context.

    Evo 2's NIM API cannot score a sequence you already have - `/forward`
    returns embeddings, and there is no likelihood endpoint. So the check is
    indirect and, stated plainly: prompt the model with the gap's left flank,
    ask what it predicts follows, and measure how far each candidate agrees.

    Agreement is weighted by the model's own per-position confidence, because
    a mismatch where Evo 2 was unsure is weak evidence against a candidate,
    while one where it was certain is strong.

    This never contributes sequence. Evo 2's prediction is recorded for the run
    log and discarded: a generated base has no traceable evidence behind it,
    and every base this agent reports must be attributable to a reference.
    """

    name = "evo2_plausibility"
    description = (
        "Compare proposed gap reconstructions against what the Evo 2 genomic foundation "
        "model predicts follows the gap's left flank, weighted by the model's confidence. "
        "Use to choose between candidates that homology evidence alone cannot separate, or "
        "to flag a consensus that aligns well but contradicts what a genome model expects. "
        "Requires NVIDIA_API_KEY."
    )
    estimated_seconds = 10.0

    def __init__(self, client: Evo2Client) -> None:
        self._client = client

    async def run(self, payload: PlausibilityInput) -> PlausibilityOutput:
        if not payload.candidates:
            return PlausibilityOutput(succeeded=False, error="No candidates to score.")
        if not payload.left_flank:
            return PlausibilityOutput(
                succeeded=False, error="No left flank to prompt Evo 2 with."
            )

        # All candidates for one gap should be about the same length; predicting
        # the longest lets every candidate be compared over its own length.
        num_tokens = max(len(sequence) for sequence in payload.candidates.values())
        prompt = payload.left_flank[-payload.max_prompt_bases :]

        try:
            continuation = await self._client.generate(prompt, num_tokens=num_tokens)
        except (ReconstructionError, ValueError) as error:
            _log.warning("evo2_generate_failed", gap_id=payload.gap_id, error=str(error))
            return PlausibilityOutput(succeeded=False, error=str(error))

        agreements = [
            self._compare(candidate_id, sequence, continuation)
            for candidate_id, sequence in payload.candidates.items()
        ]
        scores = {item.candidate_id: item.weighted_agreement for item in agreements}
        best = max(scores, key=lambda key: scores[key]) if scores else None

        return PlausibilityOutput(
            succeeded=True,
            scores=scores,
            agreements=agreements,
            best_candidate=best,
            predicted_sequence=continuation.sequence,
            model_confidence=round(continuation.mean_confidence, 4),
            diagnostics={
                "gap_id": payload.gap_id,
                "prompt_bases": len(prompt),
                "predicted_bases": len(continuation.sequence),
            },
        )

    @staticmethod
    def _compare(
        candidate_id: str, candidate: str, continuation: Continuation
    ) -> CandidateAgreement:
        """Position-wise agreement between a candidate and Evo 2's prediction.

        Compared over the overlap only. A candidate longer than the prediction
        is not penalised for the excess here - length plausibility is the
        validator's job, not the model's.
        """
        predicted = continuation.sequence
        probs = continuation.sampled_probs
        overlap = min(len(candidate), len(predicted))

        if overlap == 0:
            return CandidateAgreement(
                candidate_id=candidate_id, agreement=0.0, weighted_agreement=0.0
            )

        matches = 0
        weighted_hit = 0.0
        weight_total = 0.0

        for index in range(overlap):
            confidence = probs[index] if index < len(probs) else 1.0
            weight_total += confidence
            if candidate[index] == predicted[index]:
                matches += 1
                weighted_hit += confidence

        return CandidateAgreement(
            candidate_id=candidate_id,
            agreement=round(matches / overlap, 4),
            weighted_agreement=round(weighted_hit / weight_total, 4) if weight_total else 0.0,
        )
