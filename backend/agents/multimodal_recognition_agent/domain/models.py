"""Internal domain models.

The agent has exactly one core function: take one animal image plus a non-empty
text instruction and name the most likely species. There is no vector database,
no reference-image corpus and no similarity feature, so nothing here describes a
retrieved point, a neighbour or a distance. What the classifier returns is a
ranked list of *taxonomic labels*.

Two models carry a hard safety property worth stating explicitly.

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

# The complete set of things a Recognition request can be asking for. There is
# deliberately no `similarity` member: finding visually similar animals or
# images is not a capability of this agent, and an enum value for it would be
# the first step back towards a nearest-neighbour pipeline.
Intent = Literal["recognition", "scientific_follow_up"]

TextAlignment = Literal["agree", "neutral", "conflict"]
Decision = Literal["identified", "uncertain", "not_identified"]

# How much of a candidate's taxonomy the MOCKED sources could supply.
# `mock_verified` means "the mock fixture had both identifiers". It does NOT
# mean anything was checked against a live GBIF or NCBI database - nothing here
# ever is.
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
    # not over the data URL string. This is the key the mock classifier's
    # fixture is indexed by.
    image_sha256: str
    # Stored header dimensions, before any EXIF transpose. The pixel-area guard
    # uses width * height, which a transpose leaves unchanged.
    width: int
    height: int
    byte_size: int
    filename: str


class TextEvidence(BaseModel):
    """What the instruction contributes: intent and controlled hints.

    Never a species. Text can shape the decision but cannot introduce a taxon
    the classifier did not return.

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

    # Set when the instruction asks for something this agent does not provide -
    # today, only `visual_similarity_search`. The request is still answered by
    # species classification; the field is how the response says, out loud, that
    # the other part of the question was not attempted. It is a *refusal*
    # marker, never a route into a different pipeline.
    unsupported_capability: str | None = None


class BioCLIPTaxonPrediction(BaseModel):
    """One taxonomic label produced by the BioCLIP-2 classification boundary.

    This is a *label*, not a database hit. It has no point id, no reference
    count, no dataset version and no distance, because none of those exist in a
    classifier's output. `classification_score` is the classifier's own ranking
    score for this label on this image.

    In Sprint 2 every one of these comes from `MockBioCLIP2Provider`, whose
    scores are deterministic test values - not calibrated probabilities and not
    scientific confidence.
    """

    model_config = ConfigDict(extra="forbid")

    species_id: str
    scientific_name: str
    common_name: str | None = None
    rank: Literal["species"] = "species"
    classification_score: float


class SpeciesCandidate(BaseModel):
    """One classified species, after taxonomy validation has had its turn.

    Same identity as the prediction it came from - the taxonomy sources may
    annotate a candidate, never create or replace one.
    """

    species_id: str
    scientific_name: str
    common_name: str | None = None
    rank: Literal["species"] = "species"
    classification_score: float
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
