from __future__ import annotations

from typing import Any

from schemas.common import AgentStatus
from workflows.state import TraitDiscoveryState


class WorkflowReportFormatter:
    def format(self, state: Any) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Trait Discovery Agent - Analysis Report")
        lines.append("=" * 60)
        lines.append("")

        lines.extend(self._render_input_request(state))
        lines.append("")
        lines.extend(self._render_workflow_execution(state))
        lines.append("")
        lines.extend(self._render_agent_status(state))
        lines.append("")
        lines.extend(self._render_final_interpretation(state))
        lines.append("")
        lines.extend(self._render_technical_notes())
        lines.append("")
        return "\n".join(lines)

    def _render_input_request(self, state: Any) -> list[str]:
        return [
            "Input Request",
            "-------------",
            f"Trait: {self._get(state, 'trait_name', 'n/a')}",
            f"Species: {self._get(state, 'species_name', 'n/a')}",
            f"Question: {self._get(state, 'instruction', 'n/a')}",
            f"Initial gene candidates: {', '.join(self._get(state, 'gene_list', [])) if self._get(state, 'gene_list', []) else 'none'}",
            f"Initial context: {self._format_context(self._get(state, 'context', {}))}",
        ]

    def _render_workflow_execution(self, state: Any) -> list[str]:
        lines = [
            "Workflow Execution",
            "------------------",
            "The Trait Discovery Agent orchestrated several specialized subagents to transform the",
            "initial gene candidates into a functional biological explanation.",
            "",
            "Execution steps:",
            "",
            "1. Gene Mapper Agent",
            "   Purpose:",
            "   Resolves candidate genes into Gene Ontology annotations.",
            "",
            "   Result:",
        ]
        go_annotations = self._get(state, 'go_annotations', [])
        if go_annotations:
            for annotation in go_annotations:
                lines.append(f"   - {annotation.gene_symbol} -> {annotation.go_name}")
        else:
            lines.append("   - No Gene Ontology annotations were produced.")

        lines.extend([
            "",
            "2. Functional Evidence Agent",
            "   Purpose:",
            "   Combines pathway and protein information.",
            "",
            "   Pathways identified:",
        ])
        pathway_data = self._get(state, 'pathway_data', [])
        if pathway_data:
            for pathway in pathway_data:
                lines.append(f"   - {pathway.pathway_name}")
        else:
            lines.append("   - No pathways were identified.")

        lines.extend([
            "",
            "   Proteins identified:",
        ])
        protein_data = self._get(state, 'protein_data', [])
        if protein_data:
            for protein in protein_data:
                lines.append(f"   - {protein.gene_symbol}: {protein.protein_name}")
        else:
            lines.append("   - No proteins were identified.")

        lines.extend([
            "",
            "3. Literature Support Agent",
            "   Purpose:",
            "   Finds supporting scientific evidence.",
            "",
            "   Evidence found:",
        ])
        evidence = self._get(state, 'evidence', [])
        if evidence:
            for evidence_item in evidence:
                lines.append(f"   - {evidence_item.title} ({evidence_item.year})")
        else:
            lines.append("   - No literature evidence was identified.")

        lines.extend([
            "",
            "4. Explanation Writer (LLM)",
            "   Purpose:",
            "   Synthesizes all gathered evidence into a final explanation.",
        ])
        return lines

    def _render_agent_status(self, state: Any) -> list[str]:
        return [
            "Agent Status",
            "------------",
            f"Gene Mapper: {self._status_label(self._get(state, 'gene_mapper_status'))}",
            f"Functional Evidence: {self._status_label(self._get(state, 'functional_evidence_status'))}",
            f"Literature Support: {self._status_label(self._get(state, 'literature_status'))}",
            f"Explanation Writer: {self._status_label(self._get(state, 'status'))}",
        ]

    def _render_final_interpretation(self, state: Any) -> list[str]:
        explanation = self._get(state, 'explanation', '')
        return [
            "Final Scientific Interpretation",
            "------------------------------",
            str(explanation).strip() or "No scientific interpretation was generated.",
        ]

    def _render_technical_notes(self) -> list[str]:
        return [
            "Technical Notes",
            "---------------",
            "- Framework: LangGraph",
            "- LLM provider: NVIDIA NIM",
            "- Biological agents: currently mocked for orchestration testing",
            "- LLM used for:",
            "  - capability resolution",
            "  - explanation synthesis",
            "- Agent cards: used as capability metadata",
        ]

    def _format_context(self, context: Any) -> str:
        if not context:
            return "none"
        if isinstance(context, dict):
            if "gene_list" in context and isinstance(context["gene_list"], list):
                return ", ".join(str(item) for item in context["gene_list"])
            return ", ".join(f"{k}={v}" for k, v in context.items())
        return str(context)

    def _status_label(self, status: AgentStatus | None) -> str:
        return status.value if isinstance(status, AgentStatus) else "UNKNOWN"

    def _get(self, state: Any, key: str, default: Any = None) -> Any:
        if hasattr(state, key):
            return getattr(state, key)
        if isinstance(state, dict):
            return state.get(key, default)
        return default
