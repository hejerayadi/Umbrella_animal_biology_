"""
Species Resolver — real NCBI Assembly eutils subagent
Resolves a species name to its genome assembly ID using NCBI eutils.
Never cached — always reflects NCBI's current record (see the
consolidated store-or-not matrix: cheap to refetch, must stay live).

Prefers RefSeq (GCF_) assemblies. Uses an esearch filter when available,
falls back to client-side GCF_-over-GCA_ preference when no RefSeq
assembly exists for a species.

Output shape matches schemas.outputs.SpeciesResolverOutput exactly
(assembly_id, scientific_name, common_name, confidence). The old mock
version of this module also carried a "taxonomic_group" field and a
get_all_species() helper used by subagents/visualization.py's
size_comparison scope — neither is part of that schema, and neither has
a cheap live-API equivalent (NCBI has no "list every species" call), so
both were dropped here. See visualization.py for how size_comparison
was adapted to work without them.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ._ncbi_client import ncbi_get
from ..schemas.outputs import SpeciesResolverOutput
from ..workflows.llm import get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


async def _search_taxonomy_core(query: str) -> list[dict]:
    """Search NCBI taxonomy for candidates matching a query string."""
    term = query.strip()
    if not term:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "taxonomy",
            "term": term,
            "retmode": "json",
            "retmax": 10,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "taxonomy",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    candidates = []
    for uid in uid_list:
        info = results.get(uid, {})
        candidates.append(
            {
                "tax_id": info.get("TaxId", uid),
                "scientific_name": info.get("ScientificName", ""),
                "common_name": info.get("CommonName", ""),
                "rank": info.get("Rank", ""),
            }
        )
    return candidates


async def _search_assembly_by_taxid_core(tax_id: str) -> list[dict]:
    """Search NCBI assembly for a given taxonomy ID.

    Tries the latest RefSeq filter first (retmax=1). If that returns
    zero results, falls back to an unfiltered search (retmax=20) with
    client-side GCF_-over-GCA_ preference.
    """
    filtered_term = f"txid{tax_id}[Organism:exp] AND latest_refseq[filter]"
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": filtered_term,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if uid_list:
        uid = uid_list[0]
        resp = await asyncio.to_thread(
            ncbi_get,
            {
                "path": "esummary.fcgi",
                "db": "assembly",
                "id": uid,
                "retmode": "json",
            },
        )
        data = resp.json()
        info = data.get("result", {}).get(uid, {})
        assembly_id = info.get("assemblyaccession", "")
        if assembly_id.startswith("GCF_"):
            return [
                {
                    "assembly_id": assembly_id,
                    "scientific_name": info.get("organism", ""),
                    "common_name": info.get("organism", ""),
                    "assembly_level": info.get("assemblylevel", ""),
                }
            ]

    unfiltered_term = f"txid{tax_id}[Organism:exp]"
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": unfiltered_term,
            "retmode": "json",
            "retmax": 20,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    assemblies = []
    for uid in uid_list:
        info = results.get(uid, {})
        assemblies.append(
            {
                "assembly_id": info.get("assemblyaccession", ""),
                "scientific_name": info.get("organism", ""),
                "common_name": info.get("organism", ""),
                "assembly_level": info.get("assemblylevel", ""),
            }
        )

    assemblies.sort(key=lambda x: (not x["assembly_id"].startswith("GCF_"), x["assembly_id"]))
    return assemblies


_SPECIES_RESOLVER_SYSTEM_PROMPT = (
    "You are the Species Resolver for the Genome Agent. Your job is to identify "
    "the correct genome assembly for a given species name using the available tools.\n\n"
    "Rules:\n"
    "1. ALWAYS call search_taxonomy first with the exact species name provided.\n"
    "2. If search_taxonomy returns multiple candidates, disambiguate by comparing "
    "common names and scientific names. Lower confidence if ambiguity remains.\n"
    "3. If a search returns empty results, try ONE reformulated query (e.g., add or "
    "remove qualifiers like 'asian', 'african', etc.).\n"
    "4. After identifying a candidate tax_id, call search_assembly_by_taxid to find assemblies.\n"
    "5. Before submitting SpeciesResolverOutput, verify that the scientific_name in "
    "your answer matches the organism name in the assembly results.\n"
    "6. If no assembly is found, submit assembly_id=null, confidence=0.0, and an "
    "honest reasoning note.\n"
    "7. NEVER fabricate an assembly_id. Only submit an assembly_id that literally "
    "appears in a search_assembly_by_taxid result.\n"
)


@tool
async def search_taxonomy(query: str) -> list[dict]:
    """Search NCBI taxonomy for candidates matching a query string."""
    return await _search_taxonomy_core(query)


@tool
async def search_assembly_by_taxid(tax_id: str) -> list[dict]:
    """Search NCBI assembly for a given taxonomy ID."""
    return await _search_assembly_by_taxid_core(tax_id)


async def resolve_species_llm(species_name: str) -> dict | None:
    """Use the LLM with tool calling to resolve a species to an assembly.

    Returns a dict matching SpeciesResolverOutput on success, or None if the
    LLM path fails or exhausts its retry budget.
    """
    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [search_taxonomy, search_assembly_by_taxid, SpeciesResolverOutput],
        tool_choice="auto",
    )

    messages: list = [
        SystemMessage(content=_SPECIES_RESOLVER_SYSTEM_PROMPT),
        HumanMessage(content=f"Resolve the species: {species_name}"),
    ]

    seen_tool_results: list[dict] = []

    for step in range(4):
        try:
            response = await asyncio.to_thread(
                invoke_with_retry,
                lambda: bound.invoke(messages),
                max_retries=1,
            )
        except Exception as exc:
            logger.info("LLM species resolver failed: %s", summarize_llm_error(exc))
            return None

        tool_calls = response.tool_calls or []
        if not tool_calls:
            continue

        messages.append(AIMessage(content="", tool_calls=tool_calls))

        for call in tool_calls:
            call_id = call["id"]
            call_name = call["name"]
            call_args = call["args"]

            if call_name == "search_taxonomy":
                try:
                    result = await search_taxonomy.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)

            elif call_name == "search_assembly_by_taxid":
                try:
                    result = await search_assembly_by_taxid.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)

            elif call_name == "SpeciesResolverOutput":
                try:
                    parsed = SpeciesResolverOutput(**call_args)
                except Exception as exc:
                    messages.append(
                        ToolMessage(
                            content=f"Error parsing output: {exc}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if parsed.assembly_id is not None:
                    grounded = any(
                        isinstance(item, dict) and item.get("assembly_id") == parsed.assembly_id
                        for item in seen_tool_results
                    )
                    if not grounded:
                        messages.append(
                            ToolMessage(
                                content=(
                                    f"Error: assembly_id '{parsed.assembly_id}' not found in any "
                                    "tool result. Please search for assemblies first and use an "
                                    "assembly_id from the results."
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        continue

                return parsed.model_dump()

    return None


async def _try_refseq_filter(species_term: str) -> tuple[str | None, str | None]:
    """Try to get the latest RefSeq assembly using an esearch filter.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism] AND latest_refseq[filter]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    uid = uid_list[0]

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})
    assembly_id = assembly_info.get("assemblyaccession", "")

    if assembly_id.startswith("GCF_"):
        return uid, assembly_id

    return None, None


async def _try_unfiltered_fallback(species_term: str) -> tuple[str | None, str | None]:
    """Fall back to unfiltered esearch, preferring GCF_ over GCA_.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 10,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    chosen_uid = None
    chosen_acc = None
    fallback_uid = None
    fallback_acc = None

    for uid in uid_list:
        info = results.get(uid, {})
        acc = info.get("assemblyaccession", "")
        if acc.startswith("GCF_") and chosen_uid is None:
            chosen_uid = uid
            chosen_acc = acc
            break
        elif acc.startswith("GCA_") and fallback_uid is None:
            fallback_uid = uid
            fallback_acc = acc

    target_uid = chosen_uid or fallback_uid
    target_acc = chosen_acc or fallback_acc

    return target_uid, target_acc


async def resolve_species(species_name: str) -> dict:
    """
    Deterministic fallback: search taxonomy first, then assemblies.
    Input: species_name (str)
    Output: dict matching SpeciesResolverOutput
            (assembly_id, scientific_name, common_name, confidence, reasoning)
    """
    key = species_name.strip()

    candidates = await _search_taxonomy_core(key)
    if candidates:
        tax_id = candidates[0].get("tax_id", "")
        scientific_name = candidates[0].get("scientific_name", "")
        common_name = candidates[0].get("common_name", "")
        assemblies = await _search_assembly_by_taxid_core(tax_id)
        if assemblies:
            assemblies.sort(key=lambda x: (not x["assembly_id"].startswith("GCF_"), x["assembly_id"]))
            chosen = assemblies[0]
            return {
                "assembly_id": chosen["assembly_id"],
                "scientific_name": scientific_name or chosen.get("scientific_name", ""),
                "common_name": common_name or chosen.get("common_name", ""),
                "confidence": 0.5,
                "reasoning": "Deterministic NCBI fallback used (no LLM)",
            }
        return {
            "assembly_id": None,
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": 0.0,
        }

    uid, assembly_id = await _try_refseq_filter(key)
    if uid is None:
        uid, assembly_id = await _try_unfiltered_fallback(key)

    if uid is None or assembly_id is None:
        return {
            "assembly_id": None,
            "scientific_name": None,
            "common_name": None,
            "confidence": 0.0,
        }

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})

    scientific_name = assembly_info.get("organism")
    common_name = assembly_info.get("organism")

    return {
        "assembly_id": assembly_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "confidence": 0.5,
        "reasoning": "Deterministic NCBI fallback used (no LLM)",
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Species Resolver live NCBI test ---")
        for species in ["tiger", "house mouse", "asian elephant", "dragon"]:
            result = await resolve_species(species)
            print(f"{species}: {result}")

    asyncio.run(_quick_test())
