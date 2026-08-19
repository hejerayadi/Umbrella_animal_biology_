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
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ._ncbi_client import ncbi_get
from ..schemas.outputs import SpeciesResolverOutput
from ..workflows.llm import coerce_null_sentinels, get_llm_client, invoke_with_retry, summarize_llm_error

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
    "8. ALWAYS fill in reasoning with a one-sentence explanation of your choice, "
    "even when confidence is 1.0 and the match is unambiguous (e.g. 'search_taxonomy "
    "returned a single exact match and search_assembly_by_taxid returned its "
    "RefSeq assembly'). Never leave reasoning empty.\n"
    "9. If a tool call you already made returns the same result again, do not "
    "repeat it verbatim — either try a genuinely different query, move to the "
    "next tool, or submit assembly_id=null with an honest reasoning note.\n"
)


@tool
async def search_taxonomy(query: str) -> list[dict]:
    """Search NCBI taxonomy for candidates matching a query string."""
    return await _search_taxonomy_core(query)


@tool
async def search_assembly_by_taxid(tax_id: str) -> list[dict]:
    """Search NCBI assembly for a given taxonomy ID."""
    return await _search_assembly_by_taxid_core(tax_id)


def _check_duplicate_tool_call(
    messages: list,
    call_name: str,
    call_args: dict,
) -> tuple[bool, str | None]:
    """Check if this tool call duplicates a previous one in the message history.

    Same guard pattern as genome_metadata.py's resolver. Without this, a model
    that gets an empty result from search_taxonomy has nothing stopping it from
    calling search_taxonomy with the *same* query again on the next step —
    which is exactly the "asain elefant" x4 loop seen in the wild. On a repeat,
    we hand back the cached result instead of re-invoking the tool, and nudge
    the model to change its approach.

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
    if call_name in ("search_taxonomy", "search_assembly_by_taxid"):
        if call_key in seen_calls:
            return True, seen_calls[call_key]
    return False, None


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
    # Distinct search_taxonomy query strings tried so far (used to require
    # a genuine reformulation, not just a repeat, before a null submission
    # is accepted — see the premature-null guard below).
    taxonomy_queries_tried: set[str] = set()
    # True the moment any search_taxonomy call has returned at least one
    # candidate. Used together with assembly_lookup_attempted below to
    # catch the "found candidates, never looked up an assembly, gave up
    # anyway" shortcut a weaker model is prone to taking.
    any_taxonomy_hit = False
    # True the moment any single search_taxonomy call has returned MORE
    # THAN ONE candidate — i.e. a genuinely ambiguous query like "elephant"
    # (African savanna elephant, African forest elephant, Asian elephant,
    # ...). Rule 2 in the system prompt says to lower confidence when
    # ambiguity remains, but that's only a suggestion the model can (and,
    # in a live run, did) ignore — it settled on one candidate, found a
    # real RefSeq assembly for it, and reported confidence=1.0 with
    # reasoning claiming "a single exact match", directly contradicting
    # its own earlier multi-candidate search_taxonomy result. This flag
    # lets the code catch that contradiction instead of trusting the
    # model's self-report.
    taxonomy_ambiguous = False
    assembly_lookup_attempted = False
    # Distinct tax_ids that search_assembly_by_taxid has actually been
    # called for so far (manually by the model, or via auto-escalation
    # below). Needed because "elephant"-style queries return *multiple*
    # taxonomy candidates: assembly_lookup_attempted alone only records
    # that *a* lookup happened, so once the first candidate's lookup came
    # back empty the old code treated the whole species as exhausted and
    # accepted a null submission without ever trying candidate #2 or #3.
    # Tracking which specific tax_ids were tried lets the guard below keep
    # escalating through the remaining untried candidates instead of
    # stopping after the first miss.
    attempted_tax_ids: set[str] = set()
    # Consecutive times each guard below has fired with no progress in
    # between. Live runs showed the smaller model can just re-emit the
    # identical rejected SpeciesResolverOutput call step after step,
    # ignoring the ToolMessage feedback entirely — burning the whole
    # max_steps budget on 8 copies of the same rejection rather than
    # actually calling search_assembly_by_taxid or reformulating. These
    # counters let the loop stop *asking* and start *acting* once it's
    # clear asking isn't working.
    consecutive_assembly_guard_hits = 0
    consecutive_reformulation_guard_hits = 0
    # Human-readable trace of what happened at each step, logged on
    # exhaustion so a silent non-convergence (as opposed to an API error)
    # is actually diagnosable instead of just "SKIP" — see the log call
    # after the loop below.
    step_trace: list[str] = []

    # 8 steps, not 6: bumped again after switching MODEL_NAME to the
    # smaller/less-contended meta/llama-3.1-8b-instruct (see workflows/llm.py
    # for why). A smaller model is more prone to needing an extra retry
    # round on a rejected/ungrounded submission or a redundant tool call
    # before it converges on the same disambiguation work the 70b model did
    # in fewer steps — this is part of the same compromise (weaker
    # reasoning, so give it more room to still land correctly rather than
    # exhausting the budget on a case like "elephant" that needs several
    # genuine round trips even for a strong model).
    max_steps = 8
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
            # WARNING, not INFO: this is the one line that explains an
            # otherwise-silent None return (a "SKIP" in the scenario
            # runner, a quiet deterministic-fallback in production). Every
            # caller of this module sets the root logger to WARNING to
            # keep other subagents' chatter out of the trace (see
            # scripts/run_species_resolver_scenarios.py) — at INFO level
            # this line never reached stderr at all, so a genuine 429 and
            # a merely-slow/misconfigured client were both indistinguishable
            # from the outside.
            logger.warning("LLM species resolver failed: %s", summarize_llm_error(exc))
            return None

        tool_calls = response.tool_calls or []
        if not tool_calls:
            step_trace.append(f"step {step + 1}: model returned no tool calls (content: {str(response.content)[:120]!r})")
            continue

        messages.append(AIMessage(content="", tool_calls=tool_calls))

        for call in tool_calls:
            call_id = call["id"]
            call_name = call["name"]
            call_args = call["args"]

            is_dup, cached = _check_duplicate_tool_call(messages, call_name, call_args)
            if is_dup:
                step_trace.append(f"step {step + 1}: repeat call to {call_name}({call_args}) intercepted")
                logger.info(
                    "[guard] repeat call to %s(%s) intercepted — returning cached result",
                    call_name,
                    call_args,
                )
                messages.append(
                    ToolMessage(
                        content=(
                            f"You already called {call_name}({call_args}) — here is that "
                            f"same result again: {cached}. Calling it again with identical "
                            "arguments will not produce a different result. Either change "
                            "your query meaningfully, move on to the next tool, or submit "
                            "SpeciesResolverOutput with assembly_id=null and confidence=0.0 "
                            "if nothing further can be found."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue

            if call_name == "search_taxonomy":
                taxonomy_queries_tried.add(json.dumps(call_args, sort_keys=True))
                try:
                    result = await search_taxonomy.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)
                    if result:
                        any_taxonomy_hit = True
                    if len(result) > 1:
                        taxonomy_ambiguous = True
                    step_trace.append(f"step {step + 1}: search_taxonomy({call_args}) -> {len(result)} result(s)")
                else:
                    step_trace.append(f"step {step + 1}: search_taxonomy({call_args}) -> {str(result)[:120]!r}")

            elif call_name == "search_assembly_by_taxid":
                assembly_lookup_attempted = True
                tax_id_arg = call_args.get("tax_id")
                if tax_id_arg is not None:
                    attempted_tax_ids.add(str(tax_id_arg))
                try:
                    result = await search_assembly_by_taxid.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)
                    step_trace.append(
                        f"step {step + 1}: search_assembly_by_taxid({call_args}) -> {len(result)} result(s)"
                    )
                else:
                    step_trace.append(
                        f"step {step + 1}: search_assembly_by_taxid({call_args}) -> {str(result)[:120]!r}"
                    )

            elif call_name == "SpeciesResolverOutput":
                try:
                    parsed = SpeciesResolverOutput(
                        **coerce_null_sentinels(
                            call_args, {"assembly_id", "scientific_name", "common_name"}
                        )
                    )
                except Exception as exc:
                    step_trace.append(f"step {step + 1}: SpeciesResolverOutput rejected — parse error: {exc}")
                    messages.append(
                        ToolMessage(
                            content=f"Error parsing output: {exc}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if parsed.assembly_id is None:
                    # Spec §4 rules 2/3 require the model to actually try
                    # before giving up: reformulate once on an empty
                    # search_taxonomy, and look up an assembly for at
                    # least one candidate before submitting null. Without
                    # this check those rules are only suggestions in the
                    # prompt — a weaker model reliably skips straight to
                    # a null submission the moment ambiguity or an empty
                    # result appears, which is exactly the "elephant"/
                    # "house mouse"/reformulation failures seen in live
                    # scenario runs. This mirrors the grounding check
                    # above: reject-and-reprompt, not a fatal error.
                    #
                    # A live run showed reject-and-reprompt alone isn't
                    # enough: the smaller model can just re-emit the exact
                    # same rejected null submission every step, ignoring
                    # the ToolMessage feedback, burning the whole
                    # max_steps budget on identical rejections instead of
                    # ever calling search_assembly_by_taxid. The two
                    # branches below escalate differently after repeated
                    # stalls, because they're not equally fixable:
                    #   - "call search_assembly_by_taxid for a candidate
                    #     you already have" is NOT a judgment call — it's
                    #     mechanical, so after 2 stalls the code just
                    #     makes that real tool call itself and hands the
                    #     model the result, rather than keep asking.
                    #   - "come up with a better spelling" genuinely does
                    #     require judgment the code can't fake, so unlike
                    #     the assembly-lookup guard above, this one never
                    #     auto-escalates or relaxes on a stall count alone.
                    #     A stall here means the model re-emitted the exact
                    #     same rejected null submission instead of making a
                    #     second search_taxonomy call — that's not "tried
                    #     and failed to reformulate", it's "didn't try", and
                    #     accepting null at that point would let the output
                    #     falsely claim a reformulation attempt that never
                    #     happened. So this guard just keeps rejecting; if
                    #     the model truly never reformulates, max_steps
                    #     exhausts and the caller's deterministic NCBI
                    #     fallback takes over instead (see resolve_species).
                    # All distinct candidate tax_ids search_taxonomy has surfaced
                    # so far, in the order first seen, minus whichever ones have
                    # already had a search_assembly_by_taxid lookup (manual or
                    # auto-escalated). If this is non-empty, the model still has
                    # unexplored candidates and a null submission is premature —
                    # this is what lets "elephant" (3 candidates) keep going past
                    # the first candidate's empty result instead of stopping there.
                    candidate_tax_ids = list(
                        dict.fromkeys(
                            str(item["tax_id"])
                            for item in seen_tool_results
                            if isinstance(item, dict) and item.get("tax_id")
                        )
                    )
                    untried_tax_ids = [t for t in candidate_tax_ids if t not in attempted_tax_ids]

                    if any_taxonomy_hit and untried_tax_ids:
                        consecutive_assembly_guard_hits += 1
                        if assembly_lookup_attempted:
                            step_trace.append(
                                f"step {step + 1}: SpeciesResolverOutput rejected — "
                                f"null submitted with {len(untried_tax_ids)} candidate(s) still "
                                "untried after an earlier lookup came back empty"
                            )
                            reject_text = (
                                f"Error: that candidate's search_assembly_by_taxid lookup found "
                                f"nothing, but {len(untried_tax_ids)} other taxonomy candidate(s) "
                                "have not been tried yet. Call search_assembly_by_taxid for "
                                f"tax_id {untried_tax_ids[0]!r} (or another untried candidate) "
                                "before submitting assembly_id=null — only give up once every "
                                "candidate's lookup has failed."
                            )
                        else:
                            step_trace.append(
                                f"step {step + 1}: SpeciesResolverOutput rejected — "
                                "null submitted without ever calling search_assembly_by_taxid"
                            )
                            reject_text = (
                                "Error: search_taxonomy returned candidate(s), but you "
                                "never called search_assembly_by_taxid for any of them. "
                                "Pick the most plausible candidate tax_id and call "
                                "search_assembly_by_taxid before submitting assembly_id=null "
                                "— only give up if that lookup also fails to find an assembly."
                            )
                        messages.append(ToolMessage(content=reject_text, tool_call_id=call_id))
                        if consecutive_assembly_guard_hits >= 2:
                            tax_id = untried_tax_ids[0]
                            auto_call_id = f"auto-{step}-{call_id}"
                            try:
                                auto_result = await search_assembly_by_taxid.ainvoke({"tax_id": tax_id})
                            except Exception as exc:
                                auto_result = f"Error: {exc}"
                            assembly_lookup_attempted = True
                            attempted_tax_ids.add(tax_id)
                            if isinstance(auto_result, list):
                                seen_tool_results.extend(auto_result)
                            step_trace.append(
                                f"step {step + 1}: auto-escalation — model stalled "
                                f"{consecutive_assembly_guard_hits}x, system called "
                                f"search_assembly_by_taxid({{'tax_id': {tax_id!r}}}) on its "
                                f"behalf -> {len(auto_result) if isinstance(auto_result, list) else 0} result(s)"
                            )
                            messages.append(
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": auto_call_id,
                                            "name": "search_assembly_by_taxid",
                                            "args": {"tax_id": tax_id},
                                            "type": "tool_call",
                                        }
                                    ],
                                )
                            )
                            messages.append(ToolMessage(content=str(auto_result), tool_call_id=auto_call_id))
                            # Reset the stall counter after an auto-escalation: the
                            # model now has a fresh result to react to, and the next
                            # rejection (if any) is against a *different* remaining
                            # candidate, not a repeat of the same stall.
                            consecutive_assembly_guard_hits = 0
                        continue
                    if not any_taxonomy_hit and len(taxonomy_queries_tried) < 2:
                        consecutive_reformulation_guard_hits += 1
                        step_trace.append(
                            f"step {step + 1}: SpeciesResolverOutput rejected — "
                            f"null submitted after only {len(taxonomy_queries_tried)} distinct "
                            f"search_taxonomy attempt(s) (stall #{consecutive_reformulation_guard_hits})"
                        )
                        reformulation_text = (
                            "Error: search_taxonomy returned no results, but you have "
                            "not yet tried a reformulated query (fix a possible typo, "
                            "drop a qualifier word, or try a synonym). Try ONE "
                            "reformulated search_taxonomy call before submitting "
                            "assembly_id=null."
                        )
                        if consecutive_reformulation_guard_hits >= 2:
                            reformulation_text += (
                                " You have resubmitted the same null result without making a "
                                "new search_taxonomy call — you must actually CALL the "
                                "search_taxonomy tool with different text, not just resubmit "
                                "SpeciesResolverOutput again."
                            )
                        messages.append(
                            ToolMessage(content=reformulation_text, tool_call_id=call_id)
                        )
                        continue

                if parsed.assembly_id is not None:
                    grounded = any(
                        isinstance(item, dict) and item.get("assembly_id") == parsed.assembly_id
                        for item in seen_tool_results
                    )
                    if not grounded:
                        step_trace.append(
                            f"step {step + 1}: SpeciesResolverOutput rejected — assembly_id "
                            f"{parsed.assembly_id!r} not grounded in any tool result"
                        )
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

                    if taxonomy_ambiguous and parsed.confidence >= 1.0:
                        step_trace.append(
                            f"step {step + 1}: SpeciesResolverOutput rejected — "
                            "confidence=1.0 submitted despite an ambiguous search_taxonomy result"
                        )
                        messages.append(
                            ToolMessage(
                                content=(
                                    "Error: search_taxonomy returned multiple candidates for this "
                                    "query, so per rule 2 this is an ambiguous case even though you "
                                    "found a valid assembly for the candidate you picked. confidence "
                                    "must be below 1.0 (e.g. 0.6-0.9 depending on how confident you "
                                    "are in your disambiguation) and reasoning must acknowledge that "
                                    "other candidates existed and explain why you picked this one. "
                                    "Resubmit SpeciesResolverOutput with an honest confidence value."
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        continue

                if not parsed.reasoning or not parsed.reasoning.strip():
                    step_trace.append(f"step {step + 1}: SpeciesResolverOutput rejected — empty reasoning")
                    messages.append(
                        ToolMessage(
                            content=(
                                "Error: reasoning must not be empty. Give a one-sentence "
                                "explanation of why this assembly_id (or null) and this "
                                "confidence are correct, then resubmit SpeciesResolverOutput."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

                return parsed.model_dump()

            else:
                step_trace.append(f"step {step + 1}: unrecognized tool call {call_name}({call_args})")

    # WARNING, not INFO — same rationale as the exception-path log above:
    # this is the other silent-None cause (loop exhausted without ever
    # producing a valid grounded submission, vs. an exception), and at
    # INFO level under this script's WARNING-only root logger it never
    # reached stderr, making a genuine non-convergence indistinguishable
    # from any other silent skip. Includes the full step_trace so a
    # non-convergence is actually diagnosable — e.g. "kept repeating the
    # same search" vs "kept submitting ungrounded ids" vs "never called
    # SpeciesResolverOutput at all" are very different problems that a
    # bare "exhausted N steps" line can't distinguish between.
    logger.warning(
        "LLM species resolver exhausted %d steps for %r without a grounded submission. Trace:\n%s",
        max_steps,
        species_name,
        "\n".join(f"  {line}" for line in step_trace) or "  (no tool calls made at all)",
    )
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