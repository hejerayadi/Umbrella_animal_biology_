"""Adapter between the HTTP boundary and the real protein workflow.

`api.py` speaks the platform contract - `instruction` plus `context` in, an
`AgentResult` out. `ProteinOrchestrator.execute` speaks the scientific one: it
wants a fully specified `ProteinAnalysisRequest` carrying a gene symbol, a
species *and* its NCBI taxonomy id. This module is the only place that knows
both, exactly as `genome_agent/orchestrator_adapter.py` does for NCBI.

The distance between the two contracts is identity resolution. The Global
Orchestrator's extractor reads the user's sentence and writes free text -
`context["gene_name"] = "TP53"`, `context["species"] = "humans"`. The workflow
needs `species.taxon_id` as a positive integer, and `capabilities/identity`
aborts the whole analysis when UniProt's organism disagrees with it. Inventing
a taxon id would fabricate the very thing the workflow then verifies, so this
module asks UniProt which protein the gene and species actually name, and
passes back the accession and taxon it was given.

Why not resolve the species on its own first: UniProt's taxonomy search ranks
by text match, so "polar bear" answers with a polar bear *adenovirus* before
`Ursus maritimus`. Searching for the protein instead lets the gene symbol
disambiguate the organism, and the matched entry then supplies an accession
and a taxon id that are true together by construction.

Nothing under `app/` is modified - only imported - so the agent's own tests,
scripts and standalone service keep working unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .app.api.v1.dependencies import get_orchestrator
from .app.configuration.settings import get_settings
from .app.contracts.agent_result import AgentResult as ScientificResult
from .app.contracts.agent_result import AgentStatus as ScientificStatus
from .app.contracts.protein_request import ProteinAnalysisRequest, ProteinTaskInput, SpeciesContract
from .app.domain.exceptions import ProteinAgentError
from .app.tools.uniprot_client import UniProtClient
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

# Enough hits to tell "one protein" from "the same gene across many species",
# without pulling a page of TrEMBL fragments for a well-studied gene.
_SEARCH_SIZE = 10

# Only the fields the identity decision actually reads. UniProt returns the
# full entry otherwise - sequences included - and the adapter throws all of
# that away, having asked the workflow to fetch the canonical entry properly.
_SEARCH_FIELDS = "accession,gene_names,organism_name,organism_id,protein_name,reviewed"

# What the Global Orchestrator's extractor calls things, most specific first.
_GENE_KEYS = ("gene_name", "resolved_gene_id", "gene_symbol", "gene")
_SPECIES_KEYS = ("species", "species_name", "scientific_name")
_ACCESSION_KEYS = ("uniprot_accession", "accession")


class IdentityUnresolved(Exception):
    """No single UniProt protein matches what the user named.

    Carries the sentence shown to the user: the caller cannot improve on it,
    because only this layer knows which of gene, species or both was the part
    UniProt could not place.
    """


@dataclass(frozen=True)
class ProteinIdentity:
    """One UniProt entry, and the taxon that entry itself reports."""

    accession: str
    gene_symbol: str
    scientific_name: str
    taxon_id: int
    reviewed: bool


def _species_candidates(species: str) -> list[str]:
    """The species as written, then its singular, for UniProt to try in turn.

    Users and the extractor both write plurals - "show me MC1R in dogs". UniProt
    stores the names a taxon is *known by*, and whether the plural is among them
    is pure luck: "humans" and "dogs" are listed, "polar bears" is not, so
    without this the same phrasing resolves for one animal and fails for the
    next.

    Singularising only the last word keeps "polar bears" -> "polar bear" while
    leaving the qualifier alone, and the result is still only ever used as an
    exact-match lookup - so a bad guess finds nothing rather than finding the
    wrong organism.
    """
    candidates = [species]
    head, _, last = species.rpartition(" ")

    # "foxes" -> "fox" before "dogs" -> "dog", since the -es rule is the more
    # specific of the two.
    singular = None
    if last.endswith("es") and len(last) > 4:
        singular = last[:-2]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        singular = last[:-1]

    if singular:
        candidates.append(f"{head} {singular}".strip())
    return candidates


def _first(context: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First non-empty string among `keys`, so callers can name a value freely."""
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entry_identity(entry: dict[str, Any], fallback_gene: str) -> ProteinIdentity | None:
    """Read one UniProt search hit, or None when it lacks an accession or taxon."""
    accession = entry.get("primaryAccession")
    organism = entry.get("organism") or {}
    taxon_id = organism.get("taxonId")
    if not accession or not isinstance(taxon_id, int) or taxon_id <= 0:
        return None

    genes = entry.get("genes") or []
    gene_symbol = (genes[0].get("geneName", {}).get("value") if genes else None) or fallback_gene
    # UniProt spells the two entry types "UniProtKB reviewed (Swiss-Prot)" and
    # "UniProtKB unreviewed (TrEMBL)", so the negative has to be ruled out
    # first - "reviewed" is a substring of "unreviewed", and testing for it
    # alone marks every entry curated.
    entry_type = str(entry.get("entryType", "")).casefold()
    return ProteinIdentity(
        accession=accession,
        gene_symbol=gene_symbol,
        scientific_name=organism.get("scientificName") or "",
        taxon_id=taxon_id,
        # Swiss-Prot: manually curated, one canonical entry per gene and
        # species. That is the one worth folding.
        reviewed="reviewed" in entry_type and "unreviewed" not in entry_type,
    )


def _select(entries: list[dict[str, Any]], gene: str) -> ProteinIdentity | None:
    """Pick the canonical entry, or None when the hits span several species.

    A reviewed entry wins outright - that is Swiss-Prot's whole purpose. With
    no reviewed hit, the unreviewed ones are only usable when they agree on
    the organism; if they do not, the request named something this layer
    cannot narrow down and must not choose between.
    """
    identities = [
        identity for identity in (_entry_identity(entry, gene) for entry in entries) if identity
    ]
    if not identities:
        return None

    for identity in identities:
        if identity.reviewed:
            return identity

    if len({identity.taxon_id for identity in identities}) == 1:
        return identities[0]
    return None


class ProteinIdentityResolver:
    """Turns a gene symbol and a species name into a UniProt entry.

    Three attempts, narrowest first. Each one asks UniProt a question that can
    only be answered by a real protein record, so a miss is a miss rather than
    a plausible-looking wrong species.
    """

    def __init__(self, client: UniProtClient) -> None:
        self._client = client

    async def resolve(self, gene: str, species: str | None) -> ProteinIdentity:
        if species:
            candidates = _species_candidates(species)

            # UniProt's organism filter already understands common names:
            # "dog" resolves to Canis lupus familiaris.
            for candidate in candidates:
                identity = await self._search(
                    f'(gene_exact:{gene}) AND (organism_name:"{candidate}")'
                )
                if identity:
                    return identity

            for candidate in candidates:
                taxon_id = await self._taxon_id(candidate)
                if taxon_id:
                    # `taxonomy_id` matches descendants too, so the genus that a
                    # plural resolves to ("humans" -> Homo) still reaches the
                    # species underneath it.
                    identity = await self._search(
                        f"(gene_exact:{gene}) AND (taxonomy_id:{taxon_id})"
                    )
                    if identity:
                        return identity

            raise IdentityUnresolved(
                f"UniProt has no protein recorded for the gene '{gene}' in '{species}'. "
                "Check the gene symbol, or name the species with its scientific name."
            )

        # No species named. A reviewed-only search is the one case where the
        # gene alone is allowed to choose the organism, and `_select` still
        # refuses when the reviewed hits disagree.
        identity = await self._search(f"(gene_exact:{gene}) AND (reviewed:true)")
        if identity:
            return identity
        raise IdentityUnresolved(
            f"The gene '{gene}' matches reviewed proteins in several species. "
            "Name the species to resolve which structure is wanted."
        )

    async def _search(self, query: str) -> ProteinIdentity | None:
        payload = await self._client.request_json(
            "GET",
            "/uniprotkb/search",
            params={
                "query": query,
                "format": "json",
                "size": _SEARCH_SIZE,
                "fields": _SEARCH_FIELDS,
            },
        )
        entries = list(payload.get("results", []))
        gene = query.split("gene_exact:", 1)[-1].split(")", 1)[0]
        return _select(entries, gene)

    async def _taxon_id(self, species: str) -> int | None:
        """The taxon whose own names contain `species`, exactly.

        Ranking by text relevance is what puts an adenovirus above the bear,
        so relevance order is ignored here: a taxon qualifies only when one of
        the names it is actually known by equals the search term.
        """
        payload = await self._client.request_json(
            "GET",
            "/taxonomy/search",
            params={"query": species, "format": "json", "size": _SEARCH_SIZE},
        )
        wanted = species.casefold()
        for record in payload.get("results", []):
            if not record.get("active", True):
                continue
            names = [record.get("scientificName"), record.get("commonName")]
            names.extend(record.get("otherNames") or [])
            if any(isinstance(name, str) and name.casefold() == wanted for name in names):
                taxon_id = record.get("taxonId")
                if isinstance(taxon_id, int) and taxon_id > 0:
                    return taxon_id
        return None


def _structure_summary(output: dict[str, Any]) -> str:
    """One sentence naming the structure that was selected, and where it came from.

    This is the `protein_structure` key the agent card promises, the string the
    Responder writes the user's answer from, and what other agents test for
    before deciding whether to escalate to this one. It stays prose because
    every one of those readers was written to expect prose.
    """
    protein = output.get("protein") or {}
    structure = output.get("selected_structure") or {}

    subject = protein.get("gene_symbol") or "the requested protein"
    if protein.get("uniprot_accession"):
        subject += f" ({protein['uniprot_accession']})"
    if protein.get("scientific_name"):
        subject += f" in {protein['scientific_name']}"

    if not structure:
        return f"No usable 3D structure was found for {subject}."

    facts = [f"{structure.get('source', 'unknown source')} {structure.get('external_id', '')}".strip()]
    if structure.get("chain_id"):
        facts.append(f"chain {structure['chain_id']}")
    if structure.get("experimental_method"):
        facts.append(str(structure["experimental_method"]))
    if structure.get("resolution_angstrom"):
        facts.append(f"{structure['resolution_angstrom']} A resolution")
    if structure.get("mean_plddt"):
        facts.append(f"mean pLDDT {structure['mean_plddt']}")
    if structure.get("sequence_coverage") is not None:
        facts.append(f"{float(structure['sequence_coverage']) * 100:.0f}% sequence coverage")

    return f"3D structure of {subject}: {', '.join(facts)}."


def _shared_context(output: dict[str, Any]) -> dict[str, Any]:
    """The keys this agent contributes to the orchestrator's shared context.

    Deliberately a summary rather than the whole `ProteinAnalysisResponse`.
    Everything a completed agent returns is merged into the context that the
    Responder renders into an LLM prompt, so the residue mappings, the
    alternative structures and the evidence list would be paid for in tokens
    on every subsequent turn while answering nothing the user asked. They stay
    in the agent's own API, which is where a client that wants them belongs.
    """
    shared: dict[str, Any] = {"protein_structure": _structure_summary(output)}

    protein = output.get("protein") or {}
    if protein.get("uniprot_accession"):
        shared["uniprot_accession"] = protein["uniprot_accession"]

    explanation = output.get("explanation") or {}
    if explanation.get("summary"):
        shared["protein_explanation"] = explanation["summary"]

    warnings = output.get("warnings") or []
    if warnings:
        shared["protein_warnings"] = list(warnings)

    # The Mol* scene, for the browser rather than the language model. Named
    # for the frontend that consumes it and skipped by the Responder's prompt
    # builder - see `_RENDER_ONLY_KEYS` in `backend/orchestrator/responder.py`.
    molstar = output.get("molstar_config") or {}
    if molstar.get("structure", {}).get("url"):
        shared["protein_viewer"] = molstar

    return shared


def to_result(scientific: ScientificResult) -> AgentResult:
    """Map the workflow's routing decision onto the platform's `AgentResult`.

    The two enums carry the same four status strings, but `CONTINUE` does not
    survive the trip. In the scientific contract it means "this was a
    transient upstream failure, run the task again"; in the platform graph it
    sends the request straight back to this same node with an unchanged
    context, which would retry the identical call until LangGraph's recursion
    limit aborts the user's whole message. A failure the Responder can explain
    is the honest version of that.
    """
    status = scientific.status
    output = scientific.output or {}

    if status is ScientificStatus.COMPLETED:
        return AgentResult(status=AgentStatus.COMPLETED, output=_shared_context(output))

    if status is ScientificStatus.NEEDS_AGENT:
        # `target_agent` is passed through but not obeyed: the workflow names
        # agents in its own vocabulary ("literature_agent"), and the platform's
        # capability resolver re-decides from the prompt text anyway.
        _logger.info("[Protein] needs another agent -> %r", scientific.prompt_to_target_agent)
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=scientific.target_agent,
            prompt_to_target_agent=scientific.prompt_to_target_agent,
            output=_shared_context(output),
        )

    if status is ScientificStatus.CONTINUE:
        reason = scientific.continuation_reason or "The protein workflow did not reach a result."
        _logger.info("[Protein] not retryable through the graph -> %s", reason)
        return AgentResult(status=AgentStatus.FAILED, output=reason)

    return AgentResult(
        status=AgentStatus.FAILED,
        output=scientific.error or "The protein workflow ended without a usable result.",
    )


class OrchestratorProteinAgent:
    """Serves the real protein workflow behind the agent's `/execute` endpoint.

    `run(request) -> AgentResult` is the whole interface `api.py` depends on.
    """

    def __init__(self) -> None:
        # All built once: the orchestrator compiles a LangGraph state machine,
        # and the client owns a pooled HTTP connection.
        self._orchestrator = get_orchestrator()
        self._uniprot = UniProtClient(get_settings())
        self._resolver = ProteinIdentityResolver(self._uniprot)

    async def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        gene = _first(context, _GENE_KEYS)
        species = _first(context, _SPECIES_KEYS)
        accession = _first(context, _ACCESSION_KEYS)

        if not gene and not accession:
            # A real dependency, unlike the mock's unconditional demand for a
            # genome: the user named a trait or a species but no gene, and
            # nothing here can fold a protein that has not been identified.
            _logger.info("[Protein] no gene or accession in context -> asking for one")
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Trait",
                prompt_to_target_agent=(
                    "Identify the gene or protein behind the requested biology and return its "
                    "gene symbol, and its UniProt accession when known, so the Protein "
                    "Visualization Agent can resolve and fold the structure."
                ),
            )

        try:
            identity = await self._resolve_identity(gene, species, accession)
        except IdentityUnresolved as exc:
            _logger.info("[Protein] identity unresolved -> %s", exc)
            return AgentResult(status=AgentStatus.FAILED, output=str(exc))
        except ProteinAgentError as exc:
            _logger.warning("[Protein] UniProt unavailable during identity resolution", exc_info=True)
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"UniProt could not be reached to identify the protein: {exc}",
            )

        _logger.info(
            "[Protein] resolved %s -> %s (%s, taxon %d)",
            gene or accession,
            identity.accession,
            identity.scientific_name,
            identity.taxon_id,
        )

        task = ProteinAnalysisRequest(
            task_id=uuid4(),
            trace_id=uuid4(),
            # Every chat turn is a fresh scientific task; the orchestrator has
            # no id of its own to borrow, and reusing one would let the
            # persistence layer treat two different questions as a repeat.
            idempotency_key=f"orchestrator-{uuid4()}",
            input=ProteinTaskInput(
                resolved_gene_id=identity.gene_symbol,
                species=SpeciesContract(
                    scientific_name=identity.scientific_name,
                    taxon_id=identity.taxon_id,
                ),
                uniprot_accession=identity.accession,
            ),
        )
        return to_result(await self._orchestrator.execute(task))

    async def _resolve_identity(
        self, gene: str | None, species: str | None, accession: str | None
    ) -> ProteinIdentity:
        """Settle on one UniProt entry before the workflow is given a task.

        An accession already in the shared context is still looked up rather
        than trusted: the workflow needs the taxon id that goes with it, and
        the entry is the only place that pairing is authoritative.
        """
        if accession:
            entry = await self._uniprot.get_entry(accession)
            identity = _entry_identity(entry, gene or accession)
            if identity:
                return identity
            raise IdentityUnresolved(
                f"UniProt returned no usable organism for accession '{accession}'."
            )

        assert gene is not None  # guaranteed by the caller's gene-or-accession check
        return await self._resolver.resolve(gene, species)
