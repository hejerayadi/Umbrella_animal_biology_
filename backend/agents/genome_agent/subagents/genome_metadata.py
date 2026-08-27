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
from ..workflows.llm import coerce_null_sentinels, get_llm_client, invoke_with_retry, summarize_llm_error

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
    "7. ALWAYS fill in reasoning with a one-sentence explanation, even when no "
    "substitution was made (e.g. 'only one assembly exists for this taxon, "
    "already RefSeq'). Never leave reasoning empty.\n"
    "8. NEVER submit an assembly_id_used that didn't literally appear in a "
    "fetch_assembly_stats or list_alternate_assemblies result.\n"
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


# Coarse ordering of NCBI assembly-level strings, used only to compare two
# *already-seen* candidates against each other — not an authoritative
# genomics ranking. "Complete Genome" here refers to the assembly-level
# label NCBI reports (e.g. for a T2T assembly), not the RefSeq/GenBank
# category.
_LEVEL_RANK = {"Complete Genome": 4, "Chromosome": 3, "Scaffold": 2, "Contig": 1}


def _assembly_rank(assembly_id: str, info: dict) -> tuple[int, int]:
    """Rank a candidate assembly for substitution purposes.

    RefSeq status (GCF_ prefix) is weighted above assembly_level: per rule
    3 in the system prompt, an authoritative RefSeq record is the intended
    "better" pick even when a newer GenBank-only assembly reports an
    equal or nominally higher assembly_level. Only among two assemblies
    with the same RefSeq status does assembly_level break the tie.
    """
    is_refseq = 1 if assembly_id.startswith("GCF_") else 0
    level = (info or {}).get("assembly_level") or ""
    return (is_refseq, _LEVEL_RANK.get(level, 0))


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

    # Assembly ids actually returned by fetch_assembly_stats / list_alternate_assemblies,
    # so a final assembly_id_used can be checked for grounding the same way
    # species_resolver.py grounds assembly_id — nothing previously stopped the
    # model from substituting an assembly_id_used it never actually looked up.
    seen_assembly_ids: set[str] = {assembly_id}
    # assembly_id -> {"assembly_level": ...} for every candidate actually
    # returned by fetch_assembly_stats or list_alternate_assemblies, so a
    # final assembly_id_used can be checked against the *best* candidate
    # that was actually seen — not just "was it grounded at all". Without
    # this, nothing stops the model from keeping a genuinely inferior
    # GenBank assembly and reporting a self-contradictory "already RefSeq"
    # reasoning for it, which is exactly the zebrafish substitution
    # failure seen live.
    seen_assembly_info: dict[str, dict] = {}
    # assembly_ids for which fetch_assembly_stats (not just
    # list_alternate_assemblies, which doesn't carry genome_size_bp /
    # chromosome_count) has actually been called — the final
    # assembly_id_used must be one of these, or the numeric stats being
    # submitted for it were never actually fetched.
    full_stats_fetched: set[str] = set()
    # True once list_alternate_assemblies has been called at least once.
    # Rule 1a requires this before a final submission; nothing previously
    # enforced it, so a model could submit right after the first
    # fetch_assembly_stats call and never learn a better assembly existed.
    alternates_checked = False
    # tax_id discovered from the first fetch_assembly_stats result, kept
    # around so a stalled model can be auto-escalated straight into
    # list_alternate_assemblies (mirrors species_resolver.py's mechanical
    # auto-escalation for a guard that has a genuinely mechanical fix).
    discovered_tax_id: str | None = None
    # Consecutive times each new guard below has fired with no progress in
    # between — same rationale and pattern as species_resolver.py's
    # consecutive_*_guard_hits: a weaker model can just re-emit the same
    # rejected GenomeMetadataOutput call instead of acting on the
    # ToolMessage feedback, burning the whole max_steps budget. Once a
    # guard is mechanically fixable (both of these are — "call this tool
    # with this argument" is not a judgment call), the code just makes
    # the call itself after 2 stalls instead of continuing to ask.
    consecutive_alternates_guard_hits = 0
    consecutive_substitution_guard_hits = 0
    # Human-readable trace of what happened at each step, logged on
    # exhaustion — see the matching mechanism in species_resolver.py's
    # resolve_species_llm for the rationale.
    step_trace: list[str] = []

    # 7 steps, not 5: same compromise as species_resolver.py's max_steps
    # bump — MODEL_NAME now defaults to the smaller/less-contended
    # meta/llama-3.1-8b-instruct, which is more prone to needing an extra
    # retry round (rejected substitution, redundant tool call) before it
    # converges on a grounded answer.
    max_steps = 7
    for step in range(max_steps):
        try:
            response = await asyncio.to_thread(
                invoke_with_retry,
                lambda: bound.invoke(messages),
                # Covers both a transient 503 capacity dip (exponential
                # backoff, up to this many attempts) and a 429 rate limit
                # (a single delayed retry, capped regardless of this
                # number) — see workflows/llm.py's invoke_with_retry /
                # _is_rate_limited_error / _is_capacity_error split.
                max_retries=4,
            )
        except Exception as exc:
            # WARNING, not INFO — see the matching comment in
            # species_resolver.py's resolve_species_llm. With the root
            # logger at WARNING (scripts/run_genome_metadata_scenarios.py),
            # this was the line that made a real 429/503 indistinguishable
            # from a merely-slow client: it never reached stderr at INFO.
            logger.warning("LLM genome metadata failed: %s", summarize_llm_error(exc))
            return None

        tool_calls = response.tool_calls or []
        if not tool_calls:
            step_trace.append(f"step {step + 1}: model returned no tool calls (content: {str(response.content)[:120]!r})")
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
                step_trace.append(f"step {step + 1}: repeat call to {call_name}({call_args}) intercepted")
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
                if isinstance(result, dict) and result.get("assembly_id"):
                    seen_assembly_ids.add(result["assembly_id"])
                    seen_assembly_info[result["assembly_id"]] = {
                        "assembly_level": result.get("assembly_level"),
                    }
                    full_stats_fetched.add(result["assembly_id"])
                    if discovered_tax_id is None and result.get("tax_id"):
                        discovered_tax_id = str(result["tax_id"])
                step_trace.append(f"step {step + 1}: fetch_assembly_stats({call_args}) -> {str(result)[:120]!r}")

            elif call_name == "list_alternate_assemblies":
                # Guard against a wrong/hallucinated tax_id satisfying the
                # "you must call list_alternate_assemblies" requirement
                # below without actually looking up the right species —
                # e.g. the model invents a tax_id before ever reading the
                # real one off the fetch_assembly_stats result, or reuses
                # a stale one. That call would "count" as having checked
                # alternates while actually returning zero (or wrong)
                # results, letting a false "only one assembly exists"
                # conclusion through untouched.
                call_tax_id = str(call_args.get("tax_id", "")).strip()
                if discovered_tax_id is not None and call_tax_id and call_tax_id != discovered_tax_id:
                    step_trace.append(
                        f"step {step + 1}: list_alternate_assemblies({call_args}) rejected — "
                        f"tax_id {call_tax_id!r} does not match {discovered_tax_id!r} from "
                        "fetch_assembly_stats"
                    )
                    messages.append(
                        ToolMessage(
                            content=(
                                f"Error: tax_id {call_tax_id!r} does not match the tax_id "
                                f"({discovered_tax_id!r}) returned by fetch_assembly_stats for "
                                f"this assembly. Call list_alternate_assemblies with "
                                f"tax_id={discovered_tax_id!r} instead."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
                alternates_checked = True
                try:
                    result = await list_alternate_assemblies.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_assembly_ids.update(
                        item["assembly_id"]
                        for item in result
                        if isinstance(item, dict) and item.get("assembly_id")
                    )
                    for item in result:
                        if isinstance(item, dict) and item.get("assembly_id"):
                            # Don't clobber a fuller entry already populated
                            # by fetch_assembly_stats (which also carries
                            # numeric stats) with this leaner one.
                            seen_assembly_info.setdefault(
                                item["assembly_id"],
                                {"assembly_level": item.get("assembly_level")},
                            )
                    step_trace.append(
                        f"step {step + 1}: list_alternate_assemblies({call_args}) -> {len(result)} result(s)"
                    )
                else:
                    step_trace.append(
                        f"step {step + 1}: list_alternate_assemblies({call_args}) -> {str(result)[:120]!r}"
                    )

            elif call_name == "GenomeMetadataOutput":
                try:
                    parsed = GenomeMetadataOutput(
                        **coerce_null_sentinels(call_args, {"karyotype", "assembly_level"})
                    )
                except Exception as exc:
                    step_trace.append(f"step {step + 1}: GenomeMetadataOutput rejected — parse error: {exc}")
                    messages.append(
                        ToolMessage(
                            content=f"Error parsing output: {exc}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if parsed.genome_size_bp is not None and parsed.genome_size_bp <= 0:
                    parsed.genome_size_bp = None

                if parsed.assembly_id_used not in seen_assembly_ids:
                    step_trace.append(
                        f"step {step + 1}: GenomeMetadataOutput rejected — assembly_id_used "
                        f"{parsed.assembly_id_used!r} not grounded in any tool result"
                    )
                    messages.append(
                        ToolMessage(
                            content=(
                                f"Error: assembly_id_used '{parsed.assembly_id_used}' was never "
                                "returned by fetch_assembly_stats or list_alternate_assemblies. "
                                "Only submit an assembly_id_used that literally appeared in a "
                                "tool result."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if not alternates_checked:
                    consecutive_alternates_guard_hits += 1
                    step_trace.append(
                        f"step {step + 1}: GenomeMetadataOutput rejected — "
                        "submitted without ever calling list_alternate_assemblies"
                    )
                    messages.append(
                        ToolMessage(
                            content=(
                                "Error: you must call list_alternate_assemblies (using the "
                                "tax_id from fetch_assembly_stats) before submitting "
                                "GenomeMetadataOutput, so any better available assembly can "
                                "actually be considered rather than assumed."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    if consecutive_alternates_guard_hits >= 2 and discovered_tax_id is not None:
                        auto_call_id = f"auto-{step}-{call_id}"
                        try:
                            auto_result = await list_alternate_assemblies.ainvoke(
                                {"tax_id": discovered_tax_id}
                            )
                        except Exception as exc:
                            auto_result = f"Error: {exc}"
                        alternates_checked = True
                        if isinstance(auto_result, list):
                            for item in auto_result:
                                if isinstance(item, dict) and item.get("assembly_id"):
                                    seen_assembly_ids.add(item["assembly_id"])
                                    seen_assembly_info.setdefault(
                                        item["assembly_id"],
                                        {"assembly_level": item.get("assembly_level")},
                                    )
                        step_trace.append(
                            f"step {step + 1}: auto-escalation — model stalled "
                            f"{consecutive_alternates_guard_hits}x, system called "
                            f"list_alternate_assemblies({{'tax_id': {discovered_tax_id!r}}}) on its "
                            f"behalf -> {len(auto_result) if isinstance(auto_result, list) else 0} result(s)"
                        )
                        messages.append(
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": auto_call_id,
                                        "name": "list_alternate_assemblies",
                                        "args": {"tax_id": discovered_tax_id},
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                        messages.append(ToolMessage(content=str(auto_result), tool_call_id=auto_call_id))
                        consecutive_alternates_guard_hits = 0
                    continue

                if seen_assembly_info:
                    best_id = max(
                        seen_assembly_info,
                        key=lambda aid: _assembly_rank(aid, seen_assembly_info[aid]),
                    )
                    best_rank = _assembly_rank(best_id, seen_assembly_info[best_id])
                    current_rank = _assembly_rank(
                        parsed.assembly_id_used, seen_assembly_info.get(parsed.assembly_id_used, {})
                    )
                    if best_id != parsed.assembly_id_used and best_rank > current_rank:
                        consecutive_substitution_guard_hits += 1
                        step_trace.append(
                            f"step {step + 1}: GenomeMetadataOutput rejected — "
                            f"kept {parsed.assembly_id_used!r} despite better candidate {best_id!r} "
                            f"(level {seen_assembly_info[best_id].get('assembly_level')!r}) already seen"
                        )
                        messages.append(
                            ToolMessage(
                                content=(
                                    f"Error: {best_id!r} "
                                    f"(assembly_level={seen_assembly_info[best_id].get('assembly_level')!r}) "
                                    f"is a genuinely better assembly than {parsed.assembly_id_used!r} "
                                    "among the ones you have already looked up "
                                    f"({'RefSeq vs. GenBank' if best_id.startswith('GCF_') and not parsed.assembly_id_used.startswith('GCF_') else 'higher assembly level'}). "
                                    "Per rule 3, substitute to it: if you have not already called "
                                    f"fetch_assembly_stats for {best_id!r}, do so now to get its "
                                    "numeric stats, then resubmit GenomeMetadataOutput with "
                                    f"assembly_id_used={best_id!r} and reasoning explaining the substitution."
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        if consecutive_substitution_guard_hits >= 2:
                            auto_call_id = f"auto-{step}-{call_id}"
                            try:
                                auto_result = await fetch_assembly_stats.ainvoke({"assembly_id": best_id})
                            except Exception as exc:
                                auto_result = f"Error: {exc}"
                            if isinstance(auto_result, dict) and auto_result.get("assembly_id"):
                                seen_assembly_ids.add(auto_result["assembly_id"])
                                seen_assembly_info[auto_result["assembly_id"]] = {
                                    "assembly_level": auto_result.get("assembly_level"),
                                }
                                full_stats_fetched.add(auto_result["assembly_id"])
                            step_trace.append(
                                f"step {step + 1}: auto-escalation — model stalled "
                                f"{consecutive_substitution_guard_hits}x, system called "
                                f"fetch_assembly_stats({{'assembly_id': {best_id!r}}}) on its "
                                "behalf"
                            )
                            messages.append(
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": auto_call_id,
                                            "name": "fetch_assembly_stats",
                                            "args": {"assembly_id": best_id},
                                            "type": "tool_call",
                                        }
                                    ],
                                )
                            )
                            messages.append(ToolMessage(content=str(auto_result), tool_call_id=auto_call_id))
                            consecutive_substitution_guard_hits = 0
                        continue

                if parsed.assembly_id_used not in full_stats_fetched:
                    step_trace.append(
                        f"step {step + 1}: GenomeMetadataOutput rejected — "
                        f"assembly_id_used {parsed.assembly_id_used!r} has no fetch_assembly_stats "
                        "result to ground its numeric fields"
                    )
                    messages.append(
                        ToolMessage(
                            content=(
                                f"Error: you have not called fetch_assembly_stats for "
                                f"{parsed.assembly_id_used!r} yet, so its genome_size_bp / "
                                "chromosome_count cannot be grounded (list_alternate_assemblies "
                                "alone does not provide those numbers). Call fetch_assembly_stats "
                                f"for {parsed.assembly_id_used!r} first, then resubmit."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if not parsed.reasoning or not parsed.reasoning.strip():
                    step_trace.append(f"step {step + 1}: GenomeMetadataOutput rejected — empty reasoning")
                    messages.append(
                        ToolMessage(
                            content=(
                                "Error: reasoning must not be empty. Give a one-sentence "
                                "explanation (e.g. why no substitution was needed, or why "
                                "you substituted), then resubmit GenomeMetadataOutput."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

                # A successful return is otherwise completely silent — only
                # the exhaustion path (below) previously logged anything.
                # That made a run which *converges* to a wrong answer
                # (guards fired, model resubmitted the same thing, or a
                # guard simply never found grounds to object) indistinguishable
                # from a genuinely clean single-shot success. Surface the
                # trace at WARNING whenever any rejection happened along the
                # way, so "it returned an answer" and "it returned the
                # *right* answer without a fight" can actually be told apart
                # from the CLI output.
                if any("rejected" in line for line in step_trace):
                    logger.warning(
                        "LLM genome metadata for assembly %r converged after guard "
                        "rejection(s). Trace:\n%s",
                        assembly_id,
                        "\n".join(f"  {line}" for line in step_trace),
                    )
                return parsed.model_dump()

            else:
                step_trace.append(f"step {step + 1}: unrecognized tool call {call_name}({call_args})")

    # WARNING, not INFO — same rationale as species_resolver.py's matching
    # fix: loop exhaustion without a grounded submission is a silent-None
    # cause just like an exception is, and INFO never reached stderr under
    # this script's WARNING-only root logger. Includes step_trace for the
    # same reason species_resolver.py's does — "kept rejecting on ungrounded
    # ids" and "never called GenomeMetadataOutput" need different fixes.
    logger.warning(
        "LLM genome metadata exhausted %d steps for assembly %r without a grounded submission. Trace:\n%s",
        max_steps,
        assembly_id,
        "\n".join(f"  {line}" for line in step_trace) or "  (no tool calls made at all)",
    )
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