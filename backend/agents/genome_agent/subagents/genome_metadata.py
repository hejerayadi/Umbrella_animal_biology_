"""
Genome Metadata - real NCBI Assembly eutils subagent
Retrieves genome size, chromosome count, and assembly level for a
resolved assembly. Never cached - numeric, exact-match, cheap to
refetch (see the consolidated store-or-not matrix).

Output shape matches schemas.outputs.GenomeMetadataOutput exactly
(genome_size_bp, chromosome_count, karyotype, assembly_level). The old
mock version of this module also exposed get_all_genome_metadata(),
used only by subagents/visualization.py's size_comparison scope to
enumerate every species it knew about. NCBI has no "list every
assembly" call, so that helper was dropped here - see visualization.py
for how size_comparison was adapted to work without it.

karyotype is left as None: NCBI Assembly's summary stats don't carry a
karyotype string (it was only ever a made-up field in the mock DB), so
there's nothing real to put there yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ._ncbi_client import ncbi_get
from ..schemas.outputs import GenomeMetadataOutput
from ..workflows.llm import get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


def _parse_meta_stats(meta_xml: str) -> dict[str, str]:
    """Parse the <Stats> block inside NCBI Assembly esummary's 'meta' field."""
    stats: dict[str, str] = {}
    if not meta_xml:
        return stats

    try:
        m = re.search(r"<Stats>(.*?)</Stats>", meta_xml, re.DOTALL)
        if not m:
            return stats
        stats_xml = m.group(1)
        wrapped = f"<root>{stats_xml}</root>"
        root = ET.fromstring(wrapped)
        for stat in root.findall("Stat"):
            category = stat.get("category")
            if category:
                stats[category] = (stat.text or "").strip()
    except ET.ParseError as exc:
        logger.warning("Failed to parse assembly meta XML: %s", exc)

    return stats


async def _resolve_assembly_uid(assembly_id: str) -> str | None:
    """Convert an assembly accession (e.g. GCF_000464555.1) to a numeric NCBI UID."""
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": f"{assembly_id}[Assembly]",
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])
    return uid_list[0] if uid_list else None


async def get_genome_metadata(assembly_id: str) -> dict:
    """
    Real version of Genome Metadata.
    Input: assembly_id (str)
    Output: dict matching GenomeMetadataOutput
            (genome_size_bp, chromosome_count, karyotype, assembly_level)
    """
    uid = await _resolve_assembly_uid(assembly_id)
    if not uid:
        return {
            "genome_size_bp": None,
            "chromosome_count": None,
            "karyotype": None,
            "assembly_level": None,
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
    meta_xml = assembly_info.get("meta", "")
    stats = _parse_meta_stats(meta_xml)

    genome_size_bp = stats.get("total_length")
    if genome_size_bp is not None:
        try:
            genome_size_bp = int(genome_size_bp)
        except ValueError:
            genome_size_bp = None

    chromosome_count = stats.get("chromosome_count")
    if chromosome_count is not None:
        try:
            chromosome_count = int(chromosome_count)
        except ValueError:
            chromosome_count = None

    assembly_level = assembly_info.get("assemblystatus") or stats.get("assembly-level")

    return {
        "genome_size_bp": genome_size_bp,
        "chromosome_count": chromosome_count,
        "karyotype": None,
        "assembly_level": assembly_level,
    }


_GENOME_METADATA_SYSTEM_PROMPT = (
    "You are the Genome Metadata agent for the Genome Agent. Your job is to "
    "fetch genome statistics for a given assembly and, if a genuinely better "
    "assembly exists for the same species, substitute it.\n\n"
    "Rules:\n"
    "1. ALWAYS call fetch_assembly_stats first with the assembly_id provided.\n"
    "1a. Never call the same tool with the same arguments twice. After your "
    "first fetch_assembly_stats call, your next action must be "
    "list_alternate_assemblies using the tax_id from that result — do not "
    "call fetch_assembly_stats again with the same assembly_id.\n"
    "2. Use the tax_id from the stats result to call list_alternate_assemblies.\n"
    "3. If a genuinely better assembly exists (e.g., chromosome-level RefSeq vs. "
    "scaffold-level GenBank), fetch its stats too and substitute.\n"
    "4. Submit GenomeMetadataOutput with assembly_id_used set to the chosen "
    "assembly ID, and reasoning explaining any substitution.\n"
    "5. If only one assembly exists or the given one is already the best, "
    "submit it directly with assembly_id_used equal to the input.\n"
    "6. Never estimate missing stats - report them as null.\n"
)


@tool
async def fetch_assembly_stats(assembly_id: str) -> dict:
    """Fetch genome size, chromosome count, assembly level for one accession."""
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": assembly_id,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return {"error": f"No assembly found for {assembly_id}"}

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
    meta_xml = info.get("meta", "")
    stats = _parse_meta_stats(meta_xml)

    genome_size_bp = stats.get("total_length")
    if genome_size_bp is not None:
        try:
            genome_size_bp = int(genome_size_bp)
        except ValueError:
            genome_size_bp = None

    chromosome_count = stats.get("chromosome_count")
    if chromosome_count is not None:
        try:
            chromosome_count = int(chromosome_count)
        except ValueError:
            chromosome_count = None

    assembly_level = info.get("assemblystatus") or stats.get("assembly-level")

    if chromosome_count == 0 and assembly_level in ("Scaffold", "Contig"):
        chromosome_count = None

    return {
        "assembly_id": info.get("assemblyaccession", assembly_id),
        "scientific_name": info.get("organism", ""),
        "assembly_level": assembly_level,
        "genome_size_bp": genome_size_bp,
        "chromosome_count": chromosome_count,
        "tax_id": info.get("taxid"),
        "submission_date": info.get("submissiondate"),
    }


@tool
async def list_alternate_assemblies(tax_id: str) -> list[dict]:
    """List assembly versions for a taxon with level + submission date.

    Tries the latest RefSeq filter first (retmax=1) to ensure the best
    RefSeq assembly is always visible, then falls back to an unfiltered
    search (retmax=20) for broad context. Results are merged with the
    RefSeq entry first, deduplicated by assembly_id.
    """
    assemblies: list[dict] = []

    # Stage 1: filtered query — best RefSeq assembly first
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
    filtered_uids = data.get("esearchresult", {}).get("idlist", [])

    if filtered_uids:
        resp = await asyncio.to_thread(
            ncbi_get,
            {
                "path": "esummary.fcgi",
                "db": "assembly",
                "id": filtered_uids[0],
                "retmode": "json",
            },
        )
        data = resp.json()
        info = data.get("result", {}).get(filtered_uids[0], {})
        assemblies.append(
            {
                "assembly_id": info.get("assemblyaccession", ""),
                "assembly_level": info.get("assemblystatus", ""),
                "submission_date": info.get("submissiondate", ""),
                "organism": info.get("organism", ""),
            }
        )

    # Stage 2: unfiltered query — broad context
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
        return assemblies

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

    seen_ids = {a["assembly_id"] for a in assemblies if a["assembly_id"]}
    for uid in uid_list:
        info = results.get(uid, {})
        acc = info.get("assemblyaccession", "")
        if acc and acc not in seen_ids:
            assemblies.append(
                {
                    "assembly_id": acc,
                    "assembly_level": info.get("assemblystatus", ""),
                    "submission_date": info.get("submissiondate", ""),
                    "organism": info.get("organism", ""),
                }
            )
            seen_ids.add(acc)

    return assemblies


def _check_duplicate_tool_call(
    messages: list,
    call_name: str,
    call_args: dict,
) -> tuple[bool, str | None]:
    """Check if this tool call duplicates a previous one in the message history.

    Returns (is_duplicate, cached_result_or_None).
    """
    seen_calls: dict[tuple[str, str], str] = {}
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for prev_call in msg.tool_calls:
                prev_key = (prev_call["name"], json.dumps(prev_call["args"], sort_keys=True))
                for later_msg in messages:
                    if hasattr(later_msg, "tool_call_id") and later_msg.tool_call_id == prev_call["id"]:
                        seen_calls[prev_key] = str(later_msg.content)
                        break

    call_key = (call_name, json.dumps(call_args, sort_keys=True))
    if call_name in ("fetch_assembly_stats", "list_alternate_assemblies"):
        if call_key in seen_calls:
            return True, seen_calls[call_key]
    return False, None


async def resolve_metadata_llm(species_name: str, assembly_id: str | None = None) -> dict | None:
    """Use the LLM with tool calling to fetch genome metadata for an assembly.

    If assembly_id is None, resolves the species first via resolve_species_llm.
    Returns a dict matching GenomeMetadataOutput on success, or None if the
    LLM path fails or exhausts its retry budget.
    """
    if assembly_id is None:
        from .species_resolver import resolve_species_llm
        species = await resolve_species_llm(species_name)
        if species is None or species.get("assembly_id") is None:
            return None
        assembly_id = species["assembly_id"]

    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [fetch_assembly_stats, list_alternate_assemblies, GenomeMetadataOutput],
        tool_choice="auto",
    )

    messages: list = [
        SystemMessage(content=_GENOME_METADATA_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Fetch genome metadata for assembly {assembly_id} (species: {species_name})."
        ),
    ]

    for step in range(5):
        try:
            response = await asyncio.to_thread(
                invoke_with_retry,
                lambda: bound.invoke(messages),
                max_retries=1,
            )
        except Exception as exc:
            logger.info("LLM genome metadata failed: %s", summarize_llm_error(exc))
            return None

        tool_calls = response.tool_calls or []
        if not tool_calls:
            continue

        messages.append(AIMessage(content="", tool_calls=tool_calls))

        # Build cache of previous tool calls to detect duplicates
        seen_calls: dict[tuple[str, str], str] = {}
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for prev_call in msg.tool_calls:
                    prev_key = (prev_call["name"], json.dumps(prev_call["args"], sort_keys=True))
                    for later_msg in messages:
                        if hasattr(later_msg, "tool_call_id") and later_msg.tool_call_id == prev_call["id"]:
                            seen_calls[prev_key] = str(later_msg.content)
                            break

        for call in tool_calls:
            call_id = call["id"]
            call_name = call["name"]
            call_args = call["args"]

            is_dup, cached = _check_duplicate_tool_call(messages, call_name, call_args)
            if is_dup:
                logger.info(f"[guard] CACHE HIT — returning cached result instead of invoking {call_name}")
                messages.append(
                    ToolMessage(
                        content=(
                            f"You already called {call_name}({call_args}) — "
                            f"here is that same result again: {cached}. "
                            "Proceed to the next step; do not call this tool again with these arguments."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue
            else:
                logger.info(f"[guard] CACHE MISS — invoking {call_name}({call_args})")

            if call_name == "fetch_assembly_stats":
                try:
                    result = await fetch_assembly_stats.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

            elif call_name == "list_alternate_assemblies":
                try:
                    result = await list_alternate_assemblies.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

            elif call_name == "GenomeMetadataOutput":
                try:
                    parsed = GenomeMetadataOutput(**call_args)
                except Exception as exc:
                    messages.append(
                        ToolMessage(
                            content=f"Error parsing output: {exc}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if parsed.genome_size_bp is not None and parsed.genome_size_bp <= 0:
                    parsed.genome_size_bp = None

                return parsed.model_dump()

    return None


async def fetch_metadata_fallback(assembly_id: str) -> dict:
    """Deterministic fallback: fetch stats for the given assembly_id only.

    No substitution logic - assembly_id_used is unchanged from input.
    Per the mission spec this failure mode is NON-FATAL; record the gap
    in reasoning rather than stopping the whole request.
    """
    try:
        stats = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "genome_size_bp": None,
            "chromosome_count": None,
            "karyotype": None,
            "assembly_level": None,
            "assembly_id_used": assembly_id,
            "reasoning": f"Metadata fetch failed: {exc}",
        }

    return {
        "genome_size_bp": stats.get("genome_size_bp"),
        "chromosome_count": stats.get("chromosome_count"),
        "karyotype": None,
        "assembly_level": stats.get("assembly_level"),
        "assembly_id_used": assembly_id,
        "reasoning": "Deterministic fallback used (no LLM) - no substitution attempted",
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Genome Metadata live NCBI test ---")
        result = await get_genome_metadata("GCF_000464555.1")
        print("Tiger assembly metadata:", result)
        assert result.get("genome_size_bp") is not None, "Expected genome_size_bp from NCBI"
        assert result.get("chromosome_count") is not None, "Expected chromosome_count from NCBI"
        print("All tests passed - checkmark")

    asyncio.run(_quick_test())
