import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import invoke_with_fallback
from .agent_catalog import build_catalog_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the capability resolver of a scientific multi-agent orchestrator.\n"
    "A worker agent has paused because it needs information it cannot produce itself.\n"
    "Think step by step about what capability is actually missing before answering, "
    "but only return the final structured decision — do not show your reasoning.\n"
    "Pick exactly one agent, from the list below, able to satisfy that request.\n"
    "Use the agent name exactly as written. Never pick the waiting agent itself.\n\n"
    "Return only valid JSON with keys target_agent, prompt_to_target_agent, and reasoning.\n\n"
    "Available agents:\n{agent_catalog}"
)

USER_PROMPT = (
    "Waiting agent: {waiting_agent}\n"
    "What it needs: {need_description}\n"
    "Context gathered so far: {known_context}\n\n"
    "Decide which agent to escalate to and draft the exact request to send it."
)

class CapabilityResolution(BaseModel):
    target_agent: str = Field(description="Exact name of the agent to escalate to, copied verbatim from the catalog.")
    prompt_to_target_agent: str = Field(description="A precise, self-contained request the target agent can act on without further clarification.")
    reasoning: str = Field(description="One sentence on why this agent — internal only, not shown to the user.")


def _parse_resolution_from_text(content: str) -> CapabilityResolution | None:
    """Best-effort parser for LLM output that may include wrappers around JSON."""
    candidates: list[str] = []
    stripped = content.strip()
    if stripped:
        candidates.append(stripped)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    json_spans = re.findall(r"\{[\s\S]*?\}", stripped)
    candidates.extend(span.strip() for span in json_spans if span.strip())

    for candidate in candidates:
        try:
            return CapabilityResolution.model_validate_json(candidate)
        except Exception:
            continue
    return None


def _fallback_resolution(
    *,
    content: str,
    waiting_agent: str,
    need_description: str,
    known_context: str,
    catalog_text: str,
) -> CapabilityResolution:
    available_agents = [line[4:].strip() for line in catalog_text.splitlines() if line.startswith("### ")]
    lowered_content = content.lower()
    lowered_need = f"{need_description}\n{known_context}".lower()

    target_agent = next((agent for agent in available_agents if agent.lower() in lowered_content), None)

    if target_agent is None and (
        "literature" in lowered_need
        or "evidence" in lowered_need
        or "paper" in lowered_need
    ):
        target_agent = next((agent for agent in available_agents if "literature" in agent.lower()), None)

    if target_agent is None and (
        "gene_list" in lowered_need
        or "gene list" in lowered_need
        or "candidate gene" in lowered_need
    ):
        target_agent = next((agent for agent in available_agents if "genome" in agent.lower()), None)

    if target_agent is None:
        target_agent = available_agents[0] if available_agents else "Genome Agent"

    prompt_to_target_agent = need_description.strip()
    if known_context.strip():
        prompt_to_target_agent = (
            f"{need_description.strip()}\n\n"
            f"Known context:\n{known_context.strip()}"
        )

    return CapabilityResolution(
        target_agent=target_agent,
        prompt_to_target_agent=prompt_to_target_agent,
        reasoning="Resolver fallback: LLM output was not valid JSON; selected the best-fit agent from the live catalog.",
    )

async def resolve_capability(waiting_agent: str, need_description: str, known_context: str) -> CapabilityResolution:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
    catalog_text = build_catalog_text(exclude=waiting_agent)

    logger.info("Resolving capability for %s: %s", waiting_agent, need_description)
    response = await invoke_with_fallback(
        prompt,
        {
            "agent_catalog": catalog_text,
            "waiting_agent": waiting_agent,
            "need_description": need_description,
            "known_context": known_context,
        },
    )
    content = getattr(response, "content", str(response)).strip()
    result = _parse_resolution_from_text(content)
    if result is None:
        logger.warning(
            "Capability resolver returned non-JSON output; applying fallback selection. output=%r",
            content[:400],
        )
        result = _fallback_resolution(
            content=content,
            waiting_agent=waiting_agent,
            need_description=need_description,
            known_context=known_context,
            catalog_text=catalog_text,
        )
    logger.info("Resolved -> %s (%s)", result.target_agent, result.reasoning)
    return result