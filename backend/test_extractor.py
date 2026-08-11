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
