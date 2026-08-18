"""Estimate phylogenetic closeness between organisms.

The authoritative answer belongs to the Evolution Agent - `card.json` lists it
under `may_need` for exactly this. What lives here is the local fallback used
when delegating is not worth a round trip, plus the parsing of the Evolution
Agent's reply when it is.
"""
from __future__ import annotations

from typing import Any

# Binomial names share a genus far more informatively than they share anything
# else available without a real taxonomy lookup. These are coarse tiers, and
# they are deliberately conservative: overstating relatedness inflates
# confidence in a reconstruction.
_SAME_SPECIES = 1.0
_SAME_GENUS = 0.8
_UNRELATED = 0.3


def _binomial(name: str) -> tuple[str, str]:
    """Split a scientific name into (genus, species), lowercased."""
    parts = name.strip().lower().split()
    genus = parts[0] if parts else ""
    species = parts[1] if len(parts) > 1 else ""
    return genus, species


def heuristic_relatedness(target: str, candidate: str) -> float:
    """A crude 0..1 closeness score from the organism names alone.

    Genus-level only. It cannot tell that Loxodonta and Mammuthus are close
    relatives, which is precisely the kind of question the Evolution Agent
    exists to answer - so treat this as a floor, not an estimate.
    """
    if not target or not candidate:
        return _UNRELATED

    target_genus, target_species = _binomial(target)
    candidate_genus, candidate_species = _binomial(candidate)

    if target_genus == candidate_genus and target_species == candidate_species:
        return _SAME_SPECIES
    if target_genus and target_genus == candidate_genus:
        return _SAME_GENUS
    return _UNRELATED


def parse_evolution_agent_reply(payload: Any) -> dict[str, float]:
    """Read a relatedness map out of the Evolution Agent's `output`.

    Tolerant of shape because that agent's payload is not pinned by a shared
    schema: accepts either {organism: score} directly or a wrapper carrying a
    `relatedness` key. Anything unparseable yields an empty map, and the caller
    falls back to the heuristic.
    """
    if isinstance(payload, dict):
        candidate = payload.get("relatedness", payload)
        if isinstance(candidate, dict):
            scores: dict[str, float] = {}
            for organism, value in candidate.items():
                try:
                    scores[str(organism)] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
            return scores
    return {}
