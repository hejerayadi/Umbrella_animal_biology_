"""Routing, trait organization, and FLUX image generation for the orchestrator."""

from __future__ import annotations

from typing import Any

from .flux_client import FluxClient, FluxGenerationError
from .prompt_builder import build_visualization_prompt
from .schema import AgentRequest, AgentResult, AgentStatus

# Traits enrich a drawing, they do not gate it.
#
# This agent used to answer `needs_agent` for the Trait Discovery Agent whenever
# `traits` was missing, which made every illustration wait on a genomics
# pipeline: Trait needs a gene list, so Genome runs, so the Gene Mapper runs -
# and that mapper is still the five-gene stub in that agent's
# `subagents/gene_mapper.py`. Real NCBI symbols never intersect those five, so
# the step fails and the picture is never drawn. "Draw an Arctic fox" does not
# need a gene list to begin with.
#
# So traits are now used when they are already in context and skipped when they
# are not. To make trait discovery a prerequisite again once those subagents
# query Gene Ontology for real, return a NEEDS_AGENT result from
# `_check_routing` targeting "Trait" (the registry key from backend/registry.py,
# not the card.json display name - the capability resolver validates against
# those keys).

# Primary output key from trait_discovery_agent/card.json ("traits").
TRAIT_CONTEXT_KEYS = ("traits", "trait_summary", "trait_interpretation")
PROTEIN_CONTEXT_KEYS = (
    "protein",
    "protein_name",
    "gene",
    # What the orchestrator's extractor writes when the user names a gene in
    # their message (see backend/orchestrator/extractor.py). It is deliberately
    # not called "gene" there - "traits"/"genome" are agent OUTPUT keys, and the
    # extractor stays clear of that namespace. Without this entry a question
    # naming only a gene reaches this agent with no subject at all.
    "gene_name",
    "resolved_gene_id",
    "protein_sequence",
)
SPECIES_CONTEXT_KEYS = ("species", "organism")


def run_orchestrator_logic(
    request: AgentRequest,
    flux_client: FluxClient | None = None,
) -> AgentResult:
    routing = _check_routing(request)
    if routing is not None:
        return routing

    organized = organize_result(request)
    traits_used = extract_trait_labels(request.context)
    prompt = build_visualization_prompt(
        organized,
        instruction=request.instruction,
        context=request.context,
    )

    client = flux_client or FluxClient()
    try:
        image = client.generate_image(prompt)
    except FluxGenerationError as exc:
        return AgentResult(status=AgentStatus.FAILED, output=str(exc))

    confidence = compute_confidence_score(traits_used, organized)

    return AgentResult(
        status=AgentStatus.COMPLETED,
        output={
            "image": image,
            "traits_used": traits_used,
            "confidence_score": confidence,
        },
    )


def _check_routing(request: AgentRequest) -> AgentResult | None:
    protein = _get_protein_identifier(request.context, request.instruction)
    species = _get_species(request.context)

    # A subject is the only hard requirement - there has to be something to
    # draw. A species alone is enough, and so is a protein alone: this agent
    # illustrates whatever biology it is handed, it does not analyse proteins,
    # so demanding both would reject perfectly drawable requests.
    if protein is None and species is None:
        return AgentResult(
            status=AgentStatus.FAILED,
            output=(
                "Missing subject to illustrate. Provide a species or a protein "
                "identifier (e.g. context['species'], context['organism'], "
                "context['gene'], context['protein_sequence'])."
            ),
        )

    # No trait check here on purpose - see the note at the top of this module.
    return None


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _get_context_value(context: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = context.get(key)
        if _is_meaningful(value):
            return value
    return None


def _get_protein_identifier(context: dict[str, Any], instruction: str) -> str | None:
    value = _get_context_value(context, PROTEIN_CONTEXT_KEYS)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _get_species(context: dict[str, Any]) -> str | None:
    value = _get_context_value(context, SPECIES_CONTEXT_KEYS)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def organize_result(request: AgentRequest) -> dict[str, Any]:
    """Transform trait discovery output into flexible visualization sections."""
    sections: list[dict[str, str]] = []

    instruction = request.instruction.strip()
    if instruction:
        sections.append({"type": "user_intent", "content": instruction})

    protein = _get_protein_identifier(request.context, request.instruction)
    species = _get_species(request.context)
    if protein:
        sections.append({"type": "protein", "content": protein})
    if species:
        sections.append({"type": "species", "content": species})

    trait_payload = _get_trait_payload(request.context)
    sections.extend(_traits_to_sections(trait_payload))

    return {"visualization_input": {"sections": sections}}


def _get_trait_payload(context: dict[str, Any]) -> Any:
    for key in TRAIT_CONTEXT_KEYS:
        value = context.get(key)
        if _is_meaningful(value):
            return value
    return None


def _traits_to_sections(trait_payload: Any) -> list[dict[str, str]]:
    if trait_payload is None:
        return []

    if isinstance(trait_payload, str):
        return [{"type": "interpretation", "content": trait_payload.strip()}]

    if isinstance(trait_payload, list):
        return _list_traits_to_sections(trait_payload)

    if isinstance(trait_payload, dict):
        return _dict_traits_to_sections(trait_payload)

    return [{"type": "interpretation", "content": str(trait_payload)}]


def _list_traits_to_sections(items: list[Any]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            sections.append({"type": "trait", "content": item.strip()})
        elif isinstance(item, dict):
            section_type = str(
                item.get("type")
                or item.get("category")
                or item.get("trait_type")
                or "trait"
            )
            content = _format_trait_dict(item)
            if content:
                sections.append({"type": section_type, "content": content})
        elif item is not None:
            sections.append({"type": "trait", "content": str(item)})
    return sections


def _dict_traits_to_sections(data: dict[str, Any]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []

    for key, value in data.items():
        if not _is_meaningful(value):
            continue
        if isinstance(value, list):
            joined = "; ".join(str(v) for v in value if v is not None)
            if joined:
                sections.append({"type": str(key), "content": joined})
        elif isinstance(value, dict):
            content = _format_trait_dict(value)
            if content:
                sections.append({"type": str(key), "content": content})
        else:
            sections.append({"type": str(key), "content": str(value).strip()})

    return sections


def _format_trait_dict(item: dict[str, Any]) -> str:
    name = item.get("name") or item.get("trait") or item.get("label")
    description = (
        item.get("description")
        or item.get("content")
        or item.get("summary")
        or item.get("value")
    )
    if name and description:
        return f"{name}: {description}"
    if name:
        return str(name)
    if description:
        return str(description)
    return str(item)


def extract_trait_labels(context: dict[str, Any]) -> list[str]:
    """Flat list of trait strings for the completed output."""
    payload = _get_trait_payload(context)
    if payload is None:
        return []

    labels: list[str] = []

    if isinstance(payload, str):
        return [payload.strip()] if payload.strip() else []

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                labels.append(item.strip())
            elif isinstance(item, dict):
                label = _format_trait_dict(item)
                if label:
                    labels.append(label)
            elif item is not None:
                labels.append(str(item))
        return labels

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list):
                for entry in value:
                    if entry is not None:
                        labels.append(f"{key}: {entry}")
            elif _is_meaningful(value):
                labels.append(f"{key}: {value}")
        return labels

    return [str(payload)]


def compute_confidence_score(
    traits_used: list[str],
    organized: dict[str, Any],
) -> float:
    """Heuristic confidence based on trait coverage and section richness."""
    sections = organized.get("visualization_input", {}).get("sections", [])
    trait_count = len(traits_used)
    section_count = len(sections)

    if trait_count == 0:
        return 0.35

    base = 0.45
    trait_bonus = min(0.35, trait_count * 0.07)
    section_bonus = min(0.15, max(0, section_count - trait_count) * 0.03)

    return round(min(0.95, base + trait_bonus + section_bonus), 2)
