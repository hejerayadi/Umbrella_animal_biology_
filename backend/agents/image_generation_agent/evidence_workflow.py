"""Gather UniProt, PDB, web search, and critic evidence for protein illustrations."""

from __future__ import annotations

import logging
from typing import Callable

import requests

from .scientific_critic import ScientificCritic, scientific_critic
from .tool_schemas import EvidenceBundle, PDBStructureResult, UniProtResult, WebSearchResult
from .tools.pdb import call_pdb
from .tools.uniprot import call_uniprot
from .tools.web_search import call_web_search

logger = logging.getLogger(__name__)


def gather_protein_evidence(
    *,
    gene: str,
    species: str | None,
    instruction: str,
    session: requests.Session | None = None,
    uniprot_fn: Callable[..., UniProtResult] = call_uniprot,
    pdb_fn: Callable[..., PDBStructureResult] = call_pdb,
    web_search_fn: Callable[..., WebSearchResult] = call_web_search,
    critic_fn: Callable[..., object] | None = None,
    critic: ScientificCritic | None = None,
) -> EvidenceBundle:
    """Run UniProt → PDB → optional web search → scientific critic."""
    http = session or requests.Session()

    uniprot = uniprot_fn(gene, species, session=http)
    pdb: PDBStructureResult | None = None
    if uniprot.success and uniprot.accession:
        pdb = pdb_fn(
            uniprot.accession,
            uniprot.sequence_length,
            session=http,
        )

    web_search: WebSearchResult | None = None
    if _needs_web_search(uniprot, pdb):
        query = _build_web_search_query(gene, species, uniprot, instruction)
        web_search = web_search_fn(query, session=http)
        if not web_search.success:
            logger.info("Web search skipped or failed: %s", web_search.error)

    bundle = EvidenceBundle(
        uniprot=uniprot,
        pdb=pdb,
        web_search=web_search,
        skipped=False,
    )

    if critic_fn is not None:
        critic_result = critic_fn(
            bundle,
            gene=gene,
            species=species,
            instruction=instruction,
            critic=critic,
        )
    else:
        critic_result = scientific_critic(
            bundle,
            gene=gene,
            species=species,
            instruction=instruction,
            critic=critic,
        )

    return EvidenceBundle(
        uniprot=uniprot,
        pdb=pdb,
        web_search=web_search,
        critic=critic_result,
        skipped=False,
    )


def _needs_web_search(uniprot: UniProtResult, pdb: PDBStructureResult | None) -> bool:
    if not uniprot.success:
        return True
    if pdb is None or not pdb.found:
        return True
    return False


def _build_web_search_query(
    gene: str,
    species: str | None,
    uniprot: UniProtResult,
    instruction: str,
) -> str:
    parts = [f"{gene} protein"]
    if species:
        parts.append(species)
    if uniprot.success and uniprot.protein_name:
        parts.append(uniprot.protein_name)
    if instruction.strip():
        parts.append(instruction.strip())
    return " ".join(parts)
