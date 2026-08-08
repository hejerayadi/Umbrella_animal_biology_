"""Turn a flat list of retrieved references into ranked distinct species.

Retrieval returns reference *points*. A single species usually owns several of
them, so the raw hit list over-represents whichever species happens to have the
most stored references. Grouping first, then ranking, fixes that.

A species' score is the mean of its best N references rather than its single
best, which smooths out one unusually high hit among that species' own
references. Note what this does NOT do: a species with a single stored
reference is scored at that reference's value, so it is not penalised for being
sparsely represented. Weighting by evidence count would be a calibration
decision, and Sprint 2 deliberately makes none.
"""
from __future__ import annotations

from .models import RetrievedReference, SpeciesCandidate

# Required on every hit. A reference missing any of these cannot be ranked
# honestly - we would not know what species it belongs to, or which dataset
# produced it - so it is dropped rather than guessed at.
REQUIRED_PAYLOAD_FIELDS = ("species_id", "scientific_name", "dataset_version", "embedding_mode")


def validate_payload(payload: dict) -> bool:
    """True when a raw Qdrant payload carries every mandatory field."""
    return all(
        payload.get(field) not in (None, "") for field in REQUIRED_PAYLOAD_FIELDS
    )


def aggregate_by_species(
    references: list[RetrievedReference],
    *,
    max_references_per_species: int,
    top_k_species: int,
) -> list[SpeciesCandidate]:
    """Group references by species, score each, and return the best `top_k`.

    Taxonomy fields are left unset here - the taxonomy provider fills them in a
    later node, and inventing them at this point is exactly what the
    specification forbids.
    """

    grouped: dict[str, list[RetrievedReference]] = {}
    for reference in references:
        grouped.setdefault(reference.species_id, []).append(reference)

    candidates: list[SpeciesCandidate] = []
    for species_id, hits in grouped.items():
        ordered = sorted(hits, key=lambda hit: hit.similarity_score, reverse=True)
        best = ordered[:max_references_per_species]
        mean_score = sum(hit.similarity_score for hit in best) / len(best)

        # Prefer the highest-scoring hit's names; they all describe the same
        # species, but the top hit is the one we would cite.
        candidates.append(
            SpeciesCandidate(
                species_id=species_id,
                scientific_name=ordered[0].scientific_name,
                common_name=ordered[0].common_name,
                similarity_score=mean_score,
                # The full count, not the truncated one: "how many references
                # matched" is more useful than "how many we averaged".
                reference_count=len(hits),
                taxonomy_status="unverified",
            )
        )

    # Ties broken by species_id so the ordering is reproducible across runs.
    candidates.sort(key=lambda c: (-c.similarity_score, c.species_id))
    return candidates[:top_k_species]


def top_margin(candidates: list[SpeciesCandidate]) -> float | None:
    """Gap between the best and second-best species.

    None when there is nothing to compare. A single candidate gets a margin of
    1.0 - there is no runner-up to be confused with.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return 1.0
    return candidates[0].similarity_score - candidates[1].similarity_score
