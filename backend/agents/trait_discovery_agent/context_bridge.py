"""Fills in the identifiers the workflow needs but the orchestrator never sends.

The Global Orchestrator speaks one shape to every agent: `instruction` plus a
flat `context` dict holding whatever the agents before it published. Its
extractor seeds `species`/`trait_name`/`gene_name` from the user's sentence,
and the Genome Agent publishes `gene_list`, `species_record` and `gene_table`.

The trait workflow needs three things that are in none of that:

* `uniprot_accessions` - `{gene_symbol: accession}`. `subagents/gene_mapper`
  reads this per gene and marks the gene `unmatched` when it is absent. With
  no accessions at all, every gene is unmatched, `go_annotations` comes back
  empty, and `join_and_route_node` turns that into a hard FAILED - so without
  this step the agent fails 100% of the time no matter what the user asks.
* `tax_id` - `subagents/protein_data` reads it to scope its UniProt query to
  the right species.
* `kegg_gene_ids` - `{gene_symbol: "hsa:FGF5"}` for `subagents/pathways`.

The first two are resolved here from UniProt, which the agent already calls.
KEGG is best-effort: its REST API keys genes by organism code, there is no
tax_id -> code endpoint, and a missing entry only lands the gene in
`malformed_ids` (Pathways failing is explicitly non-fatal per
`join_and_route_node`), so an unknown organism degrades instead of failing.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from kb.sources._http_retry import request_with_retry
from kb.sources.kegg_client import _find_kegg_gene_id_raw
from kb.sources.uniprot_client import _list_uniprot_candidates_raw

_logger = logging.getLogger(__name__)

_TAXONOMY_URL = "https://rest.uniprot.org/taxonomy/search"

# Every gene costs one UniProt round trip here, then one QuickGO round trip and
# possibly one NIM call inside Gene Mapper, then the same again in Pathways and
# Protein Data. The Genome Agent caps its own `gene_list` at 50
# (`subagents/gene_annotation.py`), which at four-plus network calls each is
# several minutes - well past the orchestrator's 120s per-agent timeout in
# `worker_node.py`. Genome ranks informative genes first, so the head of the
# list is the part worth spending the budget on.
DEFAULT_GENE_LIMIT = 12


def _gene_limit() -> int:
    raw = os.getenv("TRAIT_GENE_LIMIT", "").strip()
    if not raw:
        return DEFAULT_GENE_LIMIT
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("TRAIT_GENE_LIMIT=%r is not a number; using %d", raw, DEFAULT_GENE_LIMIT)
        return DEFAULT_GENE_LIMIT
    return value if value > 0 else DEFAULT_GENE_LIMIT


# KEGG organism codes for the species this platform is asked about most. KEGG
# has no endpoint mapping an NCBI tax id to one of these, and its `link`
# endpoint needs the code as a prefix, so the common cases are listed rather
# than left to fail. `context["kegg_organism"]` overrides this for anything
# not here.
_KEGG_ORGANISM_BY_TAX_ID: dict[int, str] = {
    9606: "hsa",    # Homo sapiens
    10090: "mmu",   # Mus musculus
    10116: "rno",   # Rattus norvegicus
    9598: "ptr",    # Pan troglodytes
    9615: "cfa",    # Canis lupus familiaris
    9685: "fca",    # Felis catus
    9913: "bta",    # Bos taurus
    9823: "ssc",    # Sus scrofa
    9796: "ecb",    # Equus caballus
    9031: "gga",    # Gallus gallus
    7955: "dre",    # Danio rerio
    8364: "xtr",    # Xenopus tropicalis
    7227: "dme",    # Drosophila melanogaster
    6239: "cel",    # Caenorhabditis elegans
}


def _as_int(value: Any) -> int | None:
    """A tax id from whatever shape it arrived in, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def gene_symbols(context: dict[str, Any]) -> list[str]:
    """The candidate genes in context, de-duplicated and in their original order.

    `gene_list` is the Genome Agent's published key and the one
    `TraitDiscoveryState` already coerces. `gene_name` is what the
    orchestrator's extractor writes when the user names a single gene in their
    message, which is a perfectly good starting point for "what does FGF5 do?"
    and would otherwise be ignored.
    """
    raw = context.get("gene_list")
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    else:
        candidates = []

    if not candidates:
        single = context.get("gene_name")
        candidates = [str(single)] if single else []

    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in candidates:
        cleaned = symbol.strip()
        if cleaned and cleaned.upper() not in seen:
            seen.add(cleaned.upper())
            ordered.append(cleaned)
    return ordered


async def resolve_tax_id(context: dict[str, Any], species_name: str) -> int | None:
    """The NCBI tax id for this run, from context if any agent already knows it.

    The Genome Agent resolves the species against NCBI Taxonomy long before
    this agent runs and publishes the record under `species_record`, so asking
    UniProt again is a fallback for the case where Trait ran first.
    """
    direct = _as_int(context.get("tax_id"))
    if direct is not None:
        return direct

    record = context.get("species_record")
    if isinstance(record, dict):
        for key in ("tax_id", "taxid", "TaxId"):
            from_record = _as_int(record.get(key))
            if from_record is not None:
                return from_record

    if not species_name:
        return None
    return await _search_taxonomy(species_name)


async def _taxonomy_hits(query: str) -> list[dict[str, Any]]:
    params = {"query": query, "format": "json", "size": "10"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await request_with_retry(client, "GET", _TAXONOMY_URL, params=params)
            return response.json().get("results", [])
    except Exception as exc:  # noqa: BLE001 - a missing tax id degrades, never crashes
        _logger.warning("UniProt taxonomy lookup failed for %r: %s", query, exc)
        return []


async def _search_taxonomy(species_name: str) -> int | None:
    """The tax id for a species name, accepting only a match on the name itself.

    UniProt's taxonomy search is relevance-ranked over every name it holds, and
    the ranking is not usable as an answer on its own. Measured: an unscoped
    search for "Mus musculus" returns five hybrid and subspecies records
    without 10090 among them, and a scientific-name search for "mouse" puts
    "Mouse Mosavirus" first. Taking the top hit therefore silently scopes every
    later UniProt query to the wrong organism, which comes back as "no reviewed
    entry for this gene" rather than as an error.

    So the field-scoped queries are asked separately and the result is only
    accepted when the record's own name matches what was searched for.
    """
    wanted = species_name.strip().lower()

    scientific_hits = await _taxonomy_hits(f'scientific:"{species_name}"')
    for hit in scientific_hits:
        if str(hit.get("scientificName", "")).strip().lower() == wanted:
            return _as_int(hit.get("taxonId"))

    for hit in await _taxonomy_hits(f'common:"{species_name}"'):
        if str(hit.get("commonName", "")).strip().lower() == wanted:
            return _as_int(hit.get("taxonId"))

    # No exact hit. A binomial the user shortened or spelled slightly
    # differently still resolves here, but only against the scientific-name
    # query and only at species rank - the common-name results are dropped
    # entirely, since "mouse" legitimately prefixes a dozen unrelated species.
    for hit in scientific_hits:
        name = str(hit.get("scientificName", "")).strip().lower()
        if hit.get("rank") == "species" and name.startswith(wanted):
            _logger.info(
                "no exact taxonomy match for %r; using %s (%s)",
                species_name,
                hit.get("taxonId"),
                hit.get("scientificName"),
            )
            return _as_int(hit.get("taxonId"))

    _logger.info("UniProt taxonomy has no species named %r", species_name)
    return None


async def _accession_for(gene: str, tax_id: int) -> tuple[str, str | None]:
    """The primary reviewed UniProt accession for one gene, or None."""
    try:
        candidates = await _list_uniprot_candidates_raw(gene, tax_id)
    except Exception as exc:  # noqa: BLE001 - one gene failing must not sink the batch
        _logger.warning("UniProt lookup failed for %s (tax %d): %s", gene, tax_id, exc)
        return gene, None

    for candidate in candidates:
        accession = candidate.get("source_accession")
        if accession:
            return gene, accession
    return gene, None


async def resolve_uniprot_accessions(genes: list[str], tax_id: int) -> dict[str, str]:
    """`{gene_symbol: accession}` for every gene UniProt has a reviewed entry for.

    Genes with no reviewed entry are simply absent, which is the shape Gene
    Mapper already handles - it records them as `unmatched_genes`.
    """
    if not genes:
        return {}

    pairs = await asyncio.gather(*(_accession_for(gene, tax_id) for gene in genes))
    return {gene: accession for gene, accession in pairs if accession}


async def _kegg_id_for(organism: str, gene: str) -> tuple[str, str | None]:
    try:
        return gene, await _find_kegg_gene_id_raw(organism, gene)
    except Exception as exc:  # noqa: BLE001 - one gene failing must not sink the batch
        _logger.warning("KEGG lookup failed for %s in %s: %s", gene, organism, exc)
        return gene, None


async def resolve_kegg_gene_ids(
    genes: list[str],
    tax_id: int | None,
    context: dict[str, Any],
) -> dict[str, str]:
    """`{gene_symbol: "mmu:14176"}`, or empty when the organism is unknown.

    Genes KEGG does not know are simply absent, which is the shape the Pathways
    subagent already handles - it records them in `malformed_ids`.
    """
    organism = str(context.get("kegg_organism") or "").strip()
    if not organism and tax_id is not None:
        organism = _KEGG_ORGANISM_BY_TAX_ID.get(tax_id, "")

    if not organism:
        _logger.info(
            "no KEGG organism code for tax id %r; pathway lookup will be skipped",
            tax_id,
        )
        return {}

    pairs = await asyncio.gather(*(_kegg_id_for(organism, gene) for gene in genes))
    return {gene: kegg_id for gene, kegg_id in pairs if kegg_id}


async def prepare_workflow_context(
    context: dict[str, Any],
    species_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return the context the workflow needs, plus the genes dropped by the cap.

    The incoming context is never mutated: the orchestrator hands the same dict
    to every agent, and the identifiers resolved here are this agent's business
    only.
    """
    prepared = dict(context)
    genes = gene_symbols(context)

    limit = _gene_limit()
    dropped = genes[limit:]
    genes = genes[:limit]
    if dropped:
        _logger.info(
            "capping gene list at %d for the trait workflow; %d gene(s) not investigated",
            limit,
            len(dropped),
        )
    prepared["gene_list"] = genes

    if not genes:
        # Nothing to look identifiers up for. `check_gene_list_node` sees the
        # empty list and escalates to the Genome Agent, which is the correct
        # path - resolving a tax id first would just be a wasted round trip.
        return prepared, dropped

    tax_id = await resolve_tax_id(context, species_name)
    if tax_id is not None:
        prepared["tax_id"] = tax_id
        if not prepared.get("uniprot_accessions"):
            prepared["uniprot_accessions"] = await resolve_uniprot_accessions(genes, tax_id)
    else:
        _logger.info(
            "no tax id for %r; Gene Mapper will have no accessions to work from",
            species_name,
        )

    if not prepared.get("kegg_gene_ids"):
        prepared["kegg_gene_ids"] = await resolve_kegg_gene_ids(genes, tax_id, context)

    _logger.info(
        "prepared workflow context: %d gene(s), tax_id=%r, %d accession(s), %d KEGG id(s)",
        len(genes),
        prepared.get("tax_id"),
        len(prepared.get("uniprot_accessions") or {}),
        len(prepared.get("kegg_gene_ids") or {}),
    )
    return prepared, dropped
