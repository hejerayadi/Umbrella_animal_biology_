"""Deterministic mock for the Divergence Time worker.

Sprint 1 stub — no real TimeTree API call yet.

This mock:
- Holds known divergence times (in millions of years ago, Mya) sourced
  from TimeTree for well-studied vertebrate pairs.
- Requires exactly 2 species; returns FAILED otherwise.
- Returns FAILED for any species pair not in the catalogue, mirroring the
  "pair not in TimeTree" failure mode.
- All times are symmetric (A vs B == B vs A).

Swap this class out for ``DivergenceTimeWorker`` (Sprint 2+) without
changing the orchestrator — the ``run`` signature is stable.
"""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus

# ---------------------------------------------------------------------------
# Fixture catalogue
# Source: TimeTree of Life (Kumar et al., 2017) — widely cited values.
# ---------------------------------------------------------------------------
_CATALOGUE: set[str] = {
    "homo sapiens",
    "pan troglodytes",
    "mus musculus",
    "gallus gallus",
    "danio rerio",
}

# Key: frozenset of two species names (lowercase).
# Value: (divergence_mya, confidence)
#   confidence reflects how many calibration points TimeTree has for the pair.
_DIVERGENCE: dict[frozenset[str], tuple[float, float]] = {
    frozenset({"homo sapiens", "pan troglodytes"}): (6.0,  0.95),
    frozenset({"homo sapiens", "mus musculus"}):    (90.0, 0.92),
    frozenset({"homo sapiens", "gallus gallus"}):   (312.0, 0.88),
    frozenset({"homo sapiens", "danio rerio"}):     (450.0, 0.85),
    frozenset({"pan troglodytes", "mus musculus"}): (90.0, 0.91),
    frozenset({"pan troglodytes", "gallus gallus"}):(312.0, 0.87),
    frozenset({"pan troglodytes", "danio rerio"}):  (450.0, 0.84),
    frozenset({"mus musculus", "gallus gallus"}):   (312.0, 0.87),
    frozenset({"mus musculus", "danio rerio"}):     (450.0, 0.83),
    frozenset({"gallus gallus", "danio rerio"}):    (420.0, 0.86),
}


class DivergenceTimeMock:
    """Deterministic stand-in for the real TimeTree-backed worker."""

    def run(self, request: AgentRequest) -> AgentResult:
        species_list = self._resolve_species(request)

        if not species_list:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_list is required for divergence time estimation.",
                source_agents=["Divergence Time Agent"],
            )

        if len(species_list) != 2:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"Divergence time requires exactly 2 species; "
                    f"got {len(species_list)}: {species_list}."
                ),
                source_agents=["Divergence Time Agent"],
            )

        species_a, species_b = species_list[0], species_list[1]

        unknown = [s for s in species_list if s not in _CATALOGUE]
        if unknown:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"Species not in mock catalogue: {unknown}. "
                    "Available: " + ", ".join(sorted(_CATALOGUE)) + "."
                ),
                source_agents=["Divergence Time Agent"],
            )

        key = frozenset({species_a, species_b})
        entry = _DIVERGENCE.get(key)
        if entry is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"No divergence time record for "
                    f"'{species_a}' vs '{species_b}' in mock catalogue."
                ),
                source_agents=["Divergence Time Agent"],
            )

        divergence_mya, confidence = entry
        pair_key = f"{species_a}|{species_b}"

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "species_a": species_a,
                "species_b": species_b,
                "divergence_mya": divergence_mya,
                "confidence": confidence,
            },
            divergence_times={pair_key: divergence_mya},
            confidence=confidence,
            source_agents=["Divergence Time Agent"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_species(request: AgentRequest) -> list[str]:
        raw: list[str] = []
        if request.species_list:
            raw = request.species_list
        else:
            ctx = request.context or {}
            raw = ctx.get("species_list") or ctx.get("species") or []
            if isinstance(raw, str):
                raw = [raw]
        return [s.strip().lower() for s in raw if s.strip()]
