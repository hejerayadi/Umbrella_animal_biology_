import logging

from langchain_core.prompts import ChatPromptTemplate

from .llm import invoke_with_fallback

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a genomics research assistant summarizing findings for a scientist.\n"
    "Reason privately about how the genes, pathways, proteins, and literature relate "
    "to the trait, then write ONLY the final summary — 3-5 sentences, plain prose, "
    "no headers, no bullet points, no facts beyond what's given below."
)

USER_PROMPT = (
    "Trait: {trait_name} (species: {species_name})\n"
    "GO-annotated genes: {genes}\n"
    "Pathways: {pathways}\n"
    "Proteins: {proteins}\n"
    "Literature support: {evidence}\n"
)

async def write_explanation(trait_name, species_name, genes, pathways, proteins, evidence) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
    response = await invoke_with_fallback(
        prompt,
        {
            "trait_name": trait_name,
            "species_name": species_name,
            "genes": genes or "none matched",
            "pathways": pathways or "none found",
            "proteins": proteins or "none found",
            "evidence": evidence or "none found",
        },
        temperature=0.2,
    )
    return response.content.strip()