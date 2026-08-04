import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import get_llm
from .agent_catalog import build_catalog_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the capability resolver of a scientific multi-agent orchestrator.\n"
    "A worker agent has paused because it needs information it cannot produce itself.\n"
    "Think step by step about what capability is actually missing before answering, "
    "but only return the final structured decision — do not show your reasoning.\n"
    "Pick exactly one agent, from the list below, able to satisfy that request.\n"
    "Use the agent name exactly as written. Never pick the waiting agent itself.\n\n"
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
    llm = get_llm(temperature=0.0).with_structured_output(CapabilityResolution)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
    chain = prompt | llm

    logger.info("Resolving capability for %s: %s", waiting_agent, need_description)
    result: CapabilityResolution = await chain.ainvoke({
        "agent_catalog": build_catalog_text(exclude=waiting_agent),
        "waiting_agent": waiting_agent,
        "need_description": need_description,
        "known_context": known_context,
    })
    logger.info("Resolved -> %s (%s)", result.target_agent, result.reasoning)
    return result