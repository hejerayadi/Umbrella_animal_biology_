"""Deterministic tests for the shared-context extraction boundary."""

from backend.orchestrator.extractor import Extractor, _ExtractorOutput


class _FakeChain:
    def __init__(self, response: _ExtractorOutput) -> None:
        self.response = response

    def invoke(self, _: dict[str, str]) -> _ExtractorOutput:
        return self.response


def _extract(response: _ExtractorOutput) -> dict[str, object]:
    extractor = Extractor.__new__(Extractor)
    extractor._chain = _FakeChain(response)  # type: ignore[assignment]
    return extractor.extract("protein request")


def test_protein_specific_intent_is_added_to_shared_context() -> None:
    facts = _extract(
        _ExtractorOutput(
            species="Homo sapiens",
            gene_name="TP53",
            mutation="r273h",
            residue_position=273,
            requested_regions=[" DNA-binding domain ", ""],
            preferred_source="pdb",
            include_explanation=False,
        )
    )

    assert facts == {
        "species": "Homo sapiens",
        "gene_name": "TP53",
        "mutation": "R273H",
        "residue_position": 273,
        "requested_regions": ["DNA-binding domain"],
        "preferred_source": "pdb",
        "include_explanation": False,
    }


def test_unspecified_protein_options_are_omitted() -> None:
    assert _extract(_ExtractorOutput(species="dog")) == {"species": "dog"}


def _extract_from(query: str, response: _ExtractorOutput | None = None) -> dict[str, object]:
    """Extract from a specific message, rather than the fixed protein one.

    The sequence is found in the query itself rather than returned by the
    model, so these cases need control over the text as well as the response.
    """
    extractor = Extractor.__new__(Extractor)
    extractor._chain = _FakeChain(response or _ExtractorOutput())  # type: ignore[assignment]
    return extractor.extract(query)


def test_a_pasted_sequence_is_seeded_for_the_reconstruction_agent() -> None:
    # Without this key the Reconstruction agent has no target to repair and
    # answers FAILED, which is what the orchestrator used to do on every
    # reconstruction request.
    facts = _extract_from("Fill the gaps in ACGTACGTACGTNNNNNNNNACGTACGTACGT")

    assert facts["sequence"] == "ACGTACGTACGTNNNNNNNNACGTACGTACGT"


def test_a_wrapped_fasta_record_is_joined_into_one_sequence() -> None:
    facts = _extract_from(
        "Please repair this:\n>mammoth_frag\nACGTACGTACGTACGT\nNNNNNNNNACGTACGTA\nThanks"
    )

    # "Thanks" is spelled entirely with IUPAC ambiguity letters, so a naive
    # alphabet check appends it to the residues.
    assert facts["sequence"] == "ACGTACGTACGTACGTNNNNNNNNACGTACGTA"


def test_ordinary_prose_is_never_mistaken_for_a_sequence() -> None:
    assert "sequence" not in _extract_from("What is the genome size of the Arctic fox?")
    assert "sequence" not in _extract_from("Tell me about cattle and a gata sequence")


def test_an_accession_is_seeded_when_the_model_names_one() -> None:
    facts = _extract_from(
        "Reconstruct NC_007596", _ExtractorOutput(accession="NC_007596")
    )

    assert facts["accession"] == "NC_007596"
