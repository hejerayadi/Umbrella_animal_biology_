from backend.agents.Protein_visualization.scripts.seed_protein_knowledge import (
    _interpro_documents,
    _uniprot_documents,
)


def _entry() -> dict:  # type: ignore[type-arg]
    return {
        "primaryAccession": "P04637",
        "genes": [{"geneName": {"value": "TP53"}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}},
        "comments": [{"commentType": "FUNCTION", "texts": [{"value": "Acts as a tumor suppressor."}]}],
    }


def test_uniprot_documents_keep_real_source_identity_and_function() -> None:
    documents = _uniprot_documents(_entry())

    assert len(documents) == 2
    assert {document.source for document in documents} == {"UniProt"}
    assert {document.protein_id for document in documents} == {"P04637"}
    assert documents[1].document_type == "protein_function"


def test_interpro_documents_include_real_domain_location() -> None:
    records = [
        {
            "metadata": {"accession": "IPR011615", "name": "p53 DNA-binding domain"},
            "proteins": [{"entry_protein_locations": [{"fragments": [{"start": 102, "end": 292}]}]}],
        }
    ]

    documents = _interpro_documents(records, _entry())

    assert len(documents) == 1
    assert documents[0].source_record_id == "IPR011615"
    assert "102-292" in documents[0].text
