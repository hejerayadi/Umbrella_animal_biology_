from app.knowledge_base.retrieval import KnowledgeBase
from app.knowledge_base.schemas import KnowledgeDocument


def _document(
    document_id: str, text: str, protein_id: str = "P04637", taxonomy_id: int = 9606
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=document_id,
        text=text,
        source="UniProt",
        source_record_id=protein_id,
        protein_id=protein_id,
        gene_symbol="TP53",
        species_name="Homo sapiens",
        taxonomy_id=taxonomy_id,
        document_type="protein_function",
        section="function",
    )


async def test_local_knowledge_search_ranks_the_relevant_document() -> None:
    knowledge = KnowledgeBase()
    await knowledge.ingest(
        [
            _document("p53", "TP53 tumor suppressor DNA binding domain"),
            _document("keratin", "keratin intermediate filament assembly"),
        ]
    )

    results = await knowledge.search("TP53 DNA binding", 1, protein_id="P04637", taxonomy_id=9606)

    assert [hit.id for hit in results] == ["p53"]


async def test_search_never_widens_the_protein_or_species_filter() -> None:
    knowledge = KnowledgeBase()
    await knowledge.ingest([_document("p53", "TP53 tumor suppressor DNA binding domain")])

    assert await knowledge.search("TP53", 5, protein_id="P04637", taxonomy_id=10090) == []
    assert await knowledge.search("TP53", 5, protein_id="Q00987", taxonomy_id=9606) == []
    assert await knowledge.search("TP53", 5) == []


async def test_document_type_filter_is_applied() -> None:
    knowledge = KnowledgeBase()
    await knowledge.ingest([_document("p53", "TP53 tumor suppressor DNA binding domain")])

    hits = await knowledge.search(
        "TP53", 5, protein_id="P04637", taxonomy_id=9606, document_types=["scientific_abstract"]
    )

    assert hits == []
