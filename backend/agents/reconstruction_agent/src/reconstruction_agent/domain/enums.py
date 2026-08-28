"""Closed vocabularies shared across the agent.

Every one of these is part of a contract someone reads: the orchestrator, the
`/api/v1` client, the log stream, or a test. They are `str` enums so they
serialise to JSON as their value and compare equal to a plain string, and none
of their values may be renamed without breaking a caller.

Nothing here names an external provider. `MoleculeType` describes biology;
BLAST database codes are discovered at runtime and live in the homology layer.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable, machine-readable failure reasons.

    Clients branch on these, so the strings are the API contract; the HTTP
    status is a coarser signal that these refine.

    Note what is absent: there is no code here for "the evidence did not support
    a reconstruction". That is an answer, not a failure - see `UnresolvedReason`.
    """

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_GAP_COORDINATES = "INVALID_GAP_COORDINATES"

    ASSEMBLY_NOT_FOUND = "ASSEMBLY_NOT_FOUND"
    SEQUENCE_NOT_FOUND = "SEQUENCE_NOT_FOUND"
    SEQUENCE_PROVIDER_UNAVAILABLE = "SEQUENCE_PROVIDER_UNAVAILABLE"

    TAXONOMY_UNAVAILABLE = "TAXONOMY_UNAVAILABLE"
    DATABASE_CATALOGUE_UNAVAILABLE = "DATABASE_CATALOGUE_UNAVAILABLE"

    BLAST_SUBMISSION_FAILED = "BLAST_SUBMISSION_FAILED"
    BLAST_TIMEOUT = "BLAST_TIMEOUT"

    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"

    EVO2_UNAVAILABLE = "EVO2_UNAVAILABLE"
    EVO2_TIMEOUT = "EVO2_TIMEOUT"

    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"

    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class MoleculeType(str, Enum):
    """What kind of molecule the target sequence is.

    Biology, not provenance: it steers which reference sets can plausibly carry
    a homologue (an organelle gap is not answered by a nuclear collection) and
    it is read from the record itself, never guessed from an accession prefix.
    """

    MITOCHONDRION = "MITOCHONDRION"
    CHLOROPLAST = "CHLOROPLAST"
    PLASMID = "PLASMID"
    GENOMIC_DNA = "GENOMIC_DNA"
    UNKNOWN = "UNKNOWN"

    @property
    def is_organelle(self) -> bool:
        return self in {MoleculeType.MITOCHONDRION, MoleculeType.CHLOROPLAST}


class GapStatus(str, Enum):
    """The outcome for one requested gap."""

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    #: Never attempted - triaged out of this run, or refused up front.
    SKIPPED = "SKIPPED"


class UnresolvedReason(str, Enum):
    """Why a gap was not reconstructed.

    Refusing to reconstruct is a valid scientific result, so every one of these
    travels back to the caller with the original gap preserved. None of them is
    an error.
    """

    NO_HOMOLOGS_FOUND = "NO_HOMOLOGS_FOUND"
    #: Homologues were found, but none of them spans the missing region. This is
    #: the case that must never be mistaken for supporting evidence.
    INSUFFICIENT_GAP_SPANNING_HOMOLOGS = "INSUFFICIENT_GAP_SPANNING_HOMOLOGS"
    CONTRADICTORY_ALIGNMENTS = "CONTRADICTORY_ALIGNMENTS"
    #: Homologues that span the gap were found, and none of their sequences
    #: could be retrieved. A technical failure, not a finding about the
    #: biology - reporting it as an absence of homologues would be a false
    #: scientific claim about evidence that demonstrably exists.
    EVIDENCE_RETRIEVAL_FAILED = "EVIDENCE_RETRIEVAL_FAILED"
    CONFIDENCE_BELOW_THRESHOLD = "CONFIDENCE_BELOW_THRESHOLD"
    BIOLOGICAL_VALIDATION_FAILED = "BIOLOGICAL_VALIDATION_FAILED"
    GAP_TOO_LONG = "GAP_TOO_LONG"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    #: Triaged out: the run committed to the gaps it could finish.
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class ReconstructionStatus(str, Enum):
    """The outcome for the request as a whole.

    `PARTIALLY_COMPLETED` is load-bearing: eight resolved gaps out of ten is a
    good result, and collapsing it into `FAILED` would throw away eight
    reconstructions and mislead the orchestrator into reporting a breakdown.
    """

    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    UNRESOLVED = "UNRESOLVED"
    #: Reserved for technical failure. Not for "the evidence was insufficient".
    FAILED = "FAILED"


class ConfidenceLevel(str, Enum):
    """A coarse band over the numeric confidence, for humans."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EvaluationDecision(str, Enum):
    """What the evaluator concluded about the current state."""

    MORE_EVIDENCE_REQUIRED = "MORE_EVIDENCE_REQUIRED"
    CANDIDATE_READY = "CANDIDATE_READY"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"


class CriticDeficit(str, Enum):
    """Why the evidence is not yet sufficient.

    The critic's job is to name the deficit precisely enough that the replanner
    can act on it. A vague verdict produces a vague retry, which is how a loop
    burns its budget re-running the search that already failed.
    """

    NO_HOMOLOGS = "NO_HOMOLOGS"
    #: The searched collection does not contain the target's clade. Distinct
    #: from NO_HOMOLOGS: the homologues exist, they were not being looked for.
    DATABASE_MISMATCH = "DATABASE_MISMATCH"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    NO_GAP_SPANNING_HOMOLOG = "NO_GAP_SPANNING_HOMOLOG"
    AMBIGUOUS_ALIGNMENT = "AMBIGUOUS_ALIGNMENT"
    COMPETING_CANDIDATES = "COMPETING_CANDIDATES"
    EVO2_DISAGREEMENT = "EVO2_DISAGREEMENT"
    BIOLOGICAL_VALIDATION_FAILED = "BIOLOGICAL_VALIDATION_FAILED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class CandidateOrigin(str, Enum):
    """Where a proposed sequence came from.

    The most consequential field on a candidate. A fill assembled from twelve
    sequenced relatives and one written by a genome model are both plausible
    sequences of the same length, and nothing about the bases themselves tells
    them apart - so the distinction is carried explicitly, scored differently,
    and reported to the caller. A model prediction presented with the authority
    of an observation is the one failure this agent must never produce.
    """

    #: Assembled from references that actually span the gap. Evidence.
    HOMOLOGY = "HOMOLOGY"
    #: Written by Evo 2 continuing the flank. A prediction, not an observation:
    #: no organism is known to carry it.
    MODEL = "MODEL"


class ToolName(str, Enum):
    """The tool surface the planner may select from.

    Deliberately small. Everything else is a Python service reached through one
    of these - a planner that can call forty things spends its budget deciding
    which one, and every extra name is another way to compose an invalid plan.
    """

    GET_SEQUENCE_CONTEXT = "get_sequence_context"
    GET_ASSEMBLY_METADATA = "get_assembly_metadata"
    SEARCH_HOMOLOGS = "search_homologs"
    GET_HOMOLOG_SEQUENCES = "get_homolog_sequences"
    ALIGN_HOMOLOGS = "align_homologs"
    ANALYZE_ALIGNMENT = "analyze_alignment"
    GENERATE_CANDIDATES = "generate_candidates"
    SCORE_CANDIDATE = "score_candidate"
    EVALUATE_WITH_EVO2 = "evaluate_with_evo2"
    VALIDATE_CANDIDATE = "validate_candidate"
    RECONSTRUCT_GAP = "reconstruct_gap"
    FINALIZE_RESULT = "finalize_result"
