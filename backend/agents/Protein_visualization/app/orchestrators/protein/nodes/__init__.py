"""Workflow nodes, one module per concern.

``ProteinNodes`` is the assembled set the graph binds to. Each node is a thin
adapter: it calls exactly one capability, writes its own state keys, and turns a
provider failure into a coded warning.
"""

from dataclasses import dataclass

from backend.agents.Protein_visualization.app.capabilities.annotations import AnnotationCapability
from backend.agents.Protein_visualization.app.capabilities.critic import CriticCapability
from backend.agents.Protein_visualization.app.capabilities.evidence import EvidenceCapability
from backend.agents.Protein_visualization.app.capabilities.explanation import ExplanationCapability
from backend.agents.Protein_visualization.app.capabilities.explanation.service import LanguageModel
from backend.agents.Protein_visualization.app.capabilities.identity import IdentityCapability
from backend.agents.Protein_visualization.app.capabilities.residue_mapping import ResidueMappingCapability
from backend.agents.Protein_visualization.app.capabilities.retrieval import RetrievalCapability
from backend.agents.Protein_visualization.app.capabilities.structures import StructureCapability
from backend.agents.Protein_visualization.app.capabilities.visualization import VisualizationCapability
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.annotations import AnnotationNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.critic import CriticNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.evidence import EvidenceNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.explanation import ExplanationNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.identity import IdentityNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.mapping import ResidueMappingNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.retrieval import RetrievalNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.scene import SceneNode
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.structures import StructureNodes
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.terminal import (
    complete,
    return_needs_clarification,
    return_partial,
    scientific_abstain,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.validation import validate_input


@dataclass(frozen=True, slots=True)
class ProteinNodes:
    identity: IdentityNode
    structures: StructureNodes
    annotations: AnnotationNode
    retrieval: RetrievalNode
    residue_mapping: ResidueMappingNode
    evidence: EvidenceNode
    scene: SceneNode
    explanation: ExplanationNode
    critic: CriticNode

    @classmethod
    def build(
        cls,
        identity: IdentityCapability,
        structures: StructureCapability,
        annotations: AnnotationCapability,
        retrieval: RetrievalCapability,
        residue_mapping: ResidueMappingCapability,
        evidence: EvidenceCapability,
        visualization: VisualizationCapability,
        explanation: ExplanationCapability,
        critic: CriticCapability,
        min_sequence_coverage: float,
        llm: LanguageModel | None = None,
    ) -> "ProteinNodes":
        return cls(
            identity=IdentityNode(identity),
            structures=StructureNodes(structures, min_sequence_coverage),
            annotations=AnnotationNode(annotations),
            retrieval=RetrievalNode(retrieval),
            residue_mapping=ResidueMappingNode(residue_mapping),
            evidence=EvidenceNode(evidence),
            scene=SceneNode(visualization),
            explanation=ExplanationNode(explanation),
            critic=CriticNode(critic, llm),
        )


__all__ = [
    "ProteinNodes",
    "complete",
    "return_needs_clarification",
    "return_partial",
    "scientific_abstain",
    "validate_input",
]
