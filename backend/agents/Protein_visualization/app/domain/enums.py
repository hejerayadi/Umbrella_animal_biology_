from enum import StrEnum


class AnalysisStatus(StrEnum):
    received = "RECEIVED"
    validating = "VALIDATING"
    running = "RUNNING"
    needs_clarification = "NEEDS_CLARIFICATION"
    needs_agent = "NEEDS_AGENT"
    partial = "PARTIAL"
    completed = "COMPLETED"
    no_structure_found = "NO_STRUCTURE_FOUND"
    failed = "FAILED"


class StructureSource(StrEnum):
    pdb = "RCSB_PDB"
    alphafold = "ALPHAFOLD_DB"


class ValidationStatus(StrEnum):
    accept = "ACCEPT"
    revise = "REVISE"
    abstain = "ABSTAIN"


class PreferredSource(StrEnum):
    auto = "AUTO"
    pdb = "PDB"
    alphafold = "ALPHAFOLD"


class CapabilityName(StrEnum):
    identity = "identity"
    structures = "structures"
    annotations = "annotations"
    residue_mapping = "residue_mapping"
    retrieval = "retrieval"
    visualization = "visualization"
    explanation = "explanation"
    critic = "critic"
