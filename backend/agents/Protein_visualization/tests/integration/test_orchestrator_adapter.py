"""The boundary the Grand Orchestrator actually calls.

`orchestrator_adapter.py` sits between the platform contract
(`{instruction, context}` in, an `AgentResult` out) and the scientific one,
which needs a gene, a species *and* its taxon id. Everything here is about
that translation and the routing decisions around it; the science itself is
covered by `tests/e2e/test_workflow.py`.

UniProt is stubbed, so these run offline. The queries the stub records are the
real ones the adapter sends, which is what lets the plural and fallback tests
below assert on the order the three attempts are made in.
"""

from typing import Any

import pytest

from backend.agents.Protein_visualization.app.contracts.agent_result import (
    AgentResult as ScientificResult,
)
from backend.agents.Protein_visualization.app.contracts.agent_result import (
    AgentStatus as ScientificStatus,
)
from backend.agents.Protein_visualization.orchestrator_adapter import (
    IdentityUnresolved,
    OrchestratorProteinAgent,
    ProteinIdentityResolver,
    _select,
    _shared_context,
    _species_candidates,
    to_result,
)
from backend.agents.Protein_visualization.schema import AgentRequest, AgentStatus

REVIEWED = "UniProtKB reviewed (Swiss-Prot)"
UNREVIEWED = "UniProtKB unreviewed (TrEMBL)"


def _entry(accession: str, taxon: int, entry_type: str = REVIEWED, gene: str = "TP53") -> dict[str, Any]:
    return {
        "primaryAccession": accession,
        "entryType": entry_type,
        "genes": [{"geneName": {"value": gene}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": taxon},
    }


class StubUniProt:
    """Records every query, answers from a queue of canned payloads."""

    def __init__(self, search_results: list[list[dict[str, Any]]] | None = None) -> None:
        self._queue = list(search_results or [])
        self.queries: list[str] = []

    async def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        params = kwargs.get("params") or {}
        self.queries.append(str(params.get("query", "")))
        results = self._queue.pop(0) if self._queue else []
        return {"results": results}


# --- species candidates -------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("dogs", ["dogs", "dog"]),
        ("humans", ["humans", "human"]),
        ("polar bears", ["polar bears", "polar bear"]),
        ("foxes", ["foxes", "fox"]),
        ("mouse", ["mouse"]),
        # A binomial ending in -s gets a nonsense singular too. It costs
        # nothing: the name as written is tried first and succeeds, and the
        # candidate is only ever used as an exact-match lookup, so a bad guess
        # finds nothing rather than the wrong organism.
        ("Homo sapiens", ["Homo sapiens", "Homo sapien"]),
    ],
)
def test_plurals_get_a_singular_to_fall_back_on(written: str, expected: list[str]) -> None:
    """The extractor writes what the user typed, and users write plurals."""
    assert _species_candidates(written) == expected


def test_a_short_word_is_left_alone() -> None:
    """Stripping the 's' off "bos" or "sus" would invent a different genus."""
    assert _species_candidates("sus") == ["sus"]


# --- entry selection ----------------------------------------------------------


def test_the_reviewed_entry_wins_over_trembl_fragments() -> None:
    """A gene search for a well-studied gene always returns several entries."""
    selected = _select(
        [
            _entry("K7PPA8", 9606, UNREVIEWED),
            _entry("P04637", 9606, REVIEWED),
            _entry("E9PFT5", 9606, UNREVIEWED),
        ],
        "TP53",
    )

    assert selected is not None
    assert selected.accession == "P04637"
    assert selected.reviewed is True


def test_unreviewed_entries_are_usable_when_they_agree_on_the_organism() -> None:
    """Most species have no curated entry at all."""
    selected = _select([_entry("K7PPA8", 9606, UNREVIEWED), _entry("E9PFT5", 9606, UNREVIEWED)], "TP53")

    assert selected is not None
    assert selected.taxon_id == 9606


def test_unreviewed_entries_spanning_species_are_refused() -> None:
    """The gene named something this layer cannot narrow down."""
    assert _select([_entry("A", 9606, UNREVIEWED), _entry("B", 10090, UNREVIEWED)], "TP53") is None


def test_an_entry_without_a_taxon_is_unusable() -> None:
    entry = _entry("P04637", 9606)
    entry["organism"] = {"scientificName": "Homo sapiens"}

    assert _select([entry], "TP53") is None


# --- the three resolution attempts -------------------------------------------


async def test_the_species_is_tried_as_written_before_its_singular() -> None:
    resolver = ProteinIdentityResolver(StubUniProt([[_entry("P04637", 9606)]]))  # type: ignore[arg-type]

    identity = await resolver.resolve("TP53", "humans")

    assert identity.accession == "P04637"
    assert resolver._client.queries == ['(gene_exact:TP53) AND (organism_name:"humans")']  # type: ignore[attr-defined]


async def test_a_plural_falls_back_to_the_singular() -> None:
    """ "dogs" finds nothing; "dog" is what UniProt's organism filter knows."""
    client = StubUniProt([[], [_entry("O77616", 9615, gene="MC1R")]])
    resolver = ProteinIdentityResolver(client)  # type: ignore[arg-type]

    identity = await resolver.resolve("MC1R", "dogs")

    assert identity.taxon_id == 9615
    assert client.queries == [
        '(gene_exact:MC1R) AND (organism_name:"dogs")',
        '(gene_exact:MC1R) AND (organism_name:"dog")',
    ]


async def test_an_unresolvable_species_says_which_pair_failed() -> None:
    """The caller cannot improve on this sentence - only this layer knows."""
    resolver = ProteinIdentityResolver(StubUniProt([[], [], [], []]))  # type: ignore[arg-type]

    with pytest.raises(IdentityUnresolved, match="MC1R"):
        await resolver.resolve("MC1R", "wombles")


async def test_a_gene_with_no_species_is_only_allowed_a_reviewed_search() -> None:
    """The gene alone may choose the organism only when curation backs it."""
    client = StubUniProt([[_entry("P04637", 9606)]])
    resolver = ProteinIdentityResolver(client)  # type: ignore[arg-type]

    await resolver.resolve("TP53", None)

    assert client.queries == ["(gene_exact:TP53) AND (reviewed:true)"]


# --- routing ------------------------------------------------------------------


async def test_no_gene_and_no_accession_asks_another_agent_for_one() -> None:
    """A real dependency: nothing here can fold an unidentified protein."""
    agent = OrchestratorProteinAgent.__new__(OrchestratorProteinAgent)

    result = await agent.run(AgentRequest(instruction="what does it look like?", context={"species": "dog"}))

    assert result.status is AgentStatus.NEEDS_AGENT
    assert "gene" in (result.prompt_to_target_agent or "").lower()


def test_continue_becomes_failed_rather_than_looping_the_graph() -> None:
    """`route_after_worker` sends `continue` straight back to this same node
    with an unchanged context, which would retry the identical call until
    LangGraph's recursion limit aborts the user's whole message."""
    result = to_result(
        ScientificResult(
            status=ScientificStatus.CONTINUE,
            output={},
            continuation_reason="UniProt timed out; retry the task.",
        )
    )

    assert result.status is AgentStatus.FAILED
    assert "UniProt timed out" in str(result.output)


def test_a_completed_run_publishes_the_shared_context_keys() -> None:
    """These key names are the contract the frontend and later agents read."""
    result = to_result(
        ScientificResult(
            status=ScientificStatus.COMPLETED,
            output={
                "protein": {"uniprot_accession": "P04637", "gene_symbol": "TP53"},
                "selected_structure": {"source": "RCSB_PDB", "external_id": "1KZY"},
                "explanation": {"summary": "p53 binds DNA as a tetramer."},
                "warnings": ["CRITIC_REVISE: partial coverage."],
                "molstar_config": {"structure": {"url": "https://files.rcsb.org/1KZY.cif"}},
            },
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert set(result.output) == {
        "protein_structure",
        "uniprot_accession",
        "protein_explanation",
        "protein_warnings",
        "protein_viewer",
    }


def test_the_viewer_scene_is_only_published_when_it_can_be_rendered() -> None:
    """`protein_viewer` without a URL would give the frontend an empty canvas."""
    shared = _shared_context({"protein": {}, "molstar_config": {"structure": {}}})

    assert "protein_viewer" not in shared


def test_the_structure_summary_stays_prose() -> None:
    """The Responder writes the user's answer from this string, and other
    agents test for it before deciding whether to escalate here."""
    shared = _shared_context(
        {
            "protein": {
                "gene_symbol": "TP53",
                "uniprot_accession": "P04637",
                "scientific_name": "Homo sapiens",
            },
            "selected_structure": {
                "source": "RCSB_PDB",
                "external_id": "1KZY",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution_angstrom": 2.5,
                "sequence_coverage": 0.5,
            },
        }
    )

    summary = shared["protein_structure"]
    assert isinstance(summary, str)
    assert "TP53" in summary and "P04637" in summary and "1KZY" in summary


def test_no_structure_is_stated_rather_than_left_blank() -> None:
    shared = _shared_context({"protein": {"gene_symbol": "TP53"}, "selected_structure": {}})

    assert "No usable 3D structure" in shared["protein_structure"]
