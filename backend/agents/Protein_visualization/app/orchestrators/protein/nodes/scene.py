"""Mol* scene construction. Produces JSON only; no viewer is instantiated here."""

from typing import Any

from backend.agents.Protein_visualization.app.capabilities.visualization import VisualizationCapability
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import BUILD_SCENE
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


class SceneNode:
    def __init__(self, capability: VisualizationCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        structure = state["selected_structure"]
        with log_stage(logger, BUILD_SCENE, node=BUILD_SCENE) as outcome:
            spec = self.capability.build(structure, state["residue_mappings"])
            config = self.capability.to_config(spec, structure, state["annotations"])
            outcome["selections"] = len(config.get("selections", []))
        return executed(BUILD_SCENE, molstar_config=config)
