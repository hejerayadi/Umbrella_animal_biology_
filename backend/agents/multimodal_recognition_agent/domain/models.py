"""Internal domain models.

These follow the Sprint 2 specification's shapes. Two of them carry a hard
safety property worth stating explicitly.

`NormalizedRecognitionInput` holds the only two values in this agent that must
never escape: the decoded image bytes, and the shared context (which may contain
another agent's image output, e.g. `generated_image`). Both are declared
`Field(repr=False, exclude=True)`:

- `repr=False` keeps them out of `repr()` and `str()`;
- `exclude=True` keeps them out of `model_dump()` and `model_dump_json()`, and
  Pydantic applies it at the serializer level so a caller cannot switch it off.

That is exclusion by construction, not redaction after the fact. Stripping the
image key out of the retained context (see `validation.py`) is defence in depth
on top of it, not the guarantee itself.

Note on `frozen=True`: it prevents field *reassignment*. It does NOT make the
nested `context` dict deeply immutable - `model.context["k"] = v` still works.
We rely on nobody doing that, and on the context being a deep copy so the
caller's dict is never touched either way. It also means the model is
unhashable in practice (a dict field cannot be hashed), which is fine because
nothing hashes it.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MediaType = Literal["image/jpeg", "image/png", "image/webp"]
Intent = Literal["recognition", "similarity", "scientific_follow_up"]
TextAlignment = Literal["agree", "neutral", "conflict"]
Decision = Literal["identified", "uncertain", "not_identified"]
TaxonomyStatus = Literal["mock_verified", "partial", "unverified"]


class NormalizedRecognitionInput(BaseModel):
    """One validated, decoded image plus its instruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str

    # The original encoded JPEG/PNG/WEBP file bytes, exactly as produced by
    # strict base64 decoding. Never re-encoded, never re-compressed.
    image_bytes: bytes = Field(repr=False, exclude=True)

    # A deep copy of the caller's context with the image entry removed. Excluded
    # wholesale because it may legitimately carry other agents' image outputs.
    context: dict[str, Any] = Field(repr=False, exclude=True)

    media_type: MediaType
    # SHA-256 over `image_bytes` exactly as defined above - not over pixels and
    # not over the data URL string.
    image_sha256: str
    # Stored header dimensions, before any EXIF transpose. The pixel-area guard
    # uses width * height, which a transpose leaves unchanged.
    width: int
    height: int
    byte_size: int
    filename: str


class TextEvidence(BaseModel):
    """What the instruction contributes: intent and controlled hints.

    Never a species. Text can shape the decision but cannot introduce a
    candidate that retrieval did not return.

    `language` is a best-effort deterministic guess, `None` whenever the text
    is too short or too ambiguous to call. It is reported, never acted on - no
    branch of the workflow reads it.
    """

    intent: Intent
    language: str | None = None
    taxon_hint: str | None = None
    location_hint: str | None = None
    habitat_hint: str | None = None
    requested_capability: str | None = None


class RetrievedReference(BaseModel):
    """One reference point returned by the retrieval provider."""

    point_id: str
    species_id: str
    scientific_name: str
    common_name: str | None = None
    similarity_score: float
    source: str | None = None
    dataset_version: str
    embedding_mode: Literal["mock_bioclip2"]


class SpeciesCandidate(BaseModel):
    """Distinct species aggregated from one or more retrieved references."""

    species_id: str
    scientific_name: str
    common_name: str | None = None
    similarity_score: float
    reference_count: int
    gbif_id: int | None = None
    ncbi_taxid: int | None = None
    taxonomy_status: TaxonomyStatus


class RecognitionDecision(BaseModel):
    """The workflow's conclusion. `uncertain` and `not_identified` are
    completed scientific outcomes, not service failures."""

    decision: Decision
    primary_species: SpeciesCandidate | None
    candidates: list[SpeciesCandidate]
    text_alignment: TextAlignment
    explanation: str
    clarification_question: str | None = None
    # Set only when the visual evidence itself was insufficient. This is NOT a
    # general clarification route - it is the one narrow follow-up the validated
    # decisions allow, and it is reachable only from `not_identified`.
    request_better_image: bool = False
