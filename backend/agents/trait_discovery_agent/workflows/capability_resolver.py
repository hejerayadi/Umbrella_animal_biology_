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

async def resolve_capability(waiting_agent: str, need_description: str, known_context: str) -> CapabilityResolution:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])

    logger.info("Resolving capability for %s: %s", waiting_agent, need_description)
    response = await invoke_with_fallback(
        prompt,
        {
            "agent_catalog": build_catalog_text(exclude=waiting_agent),
            "waiting_agent": waiting_agent,
            "need_description": need_description,
            "known_context": known_context,
        },
    )
    content = getattr(response, "content", str(response)).strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    json_text = match.group(0) if match else content
    result = CapabilityResolution.model_validate_json(json_text)
    logger.info("Resolved -> %s (%s)", result.target_agent, result.reasoning)
    return result