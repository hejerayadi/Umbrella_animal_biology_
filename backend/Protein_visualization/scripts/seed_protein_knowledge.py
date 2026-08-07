"""Fetch real UniProt/InterPro records and index them into managed Qdrant."""

import argparse
import asyncio
from typing import Any

import httpx

from app.configuration.settings import get_settings
from app.knowledge_base.schemas import DocumentType, KnowledgeDocument
from app.tools import InterProClient, UniProtClient


def _identity(entry: dict[str, Any]) -> tuple[str, str, str, int, str]:
    accession = str(entry["primaryAccession"])
    genes = entry.get("genes", [])
    gene = genes[0].get("geneName", {}).get("value", accession) if genes else accession
    organism = entry.get("organism", {})
    species = str(organism.get("scientificName", "unknown species"))
    taxonomy_id = int(organism.get("taxonId", 0))
    protein_name = str(
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", accession)
    )
    return accession, str(gene), species, taxonomy_id, protein_name


def _uniprot_documents(entry: dict[str, Any]) -> list[KnowledgeDocument]:
    accession, gene, species, taxonomy_id, protein_name = _identity(entry)
    documents = [
        KnowledgeDocument(
            document_id=f"uniprot:{accession}:identity",
            text=f"UniProt {accession} identifies {gene} as {protein_name} in {species}.",
            source="UniProt",
            source_record_id=accession,
            protein_id=accession,
            gene_symbol=gene,
            species_name=species,
            taxonomy_id=taxonomy_id,
            document_type="curated_evidence",
            section="identity",
        )
    ]
    type_mapping: dict[str, DocumentType] = {
        "FUNCTION": "protein_function",
        "DOMAIN": "protein_domain",
        "SUBUNIT": "curated_evidence",
        "PTM": "curated_evidence",
        "DISEASE": "curated_evidence",
    }
    for comment_index, comment in enumerate(entry.get("comments", [])):
        comment_type = str(comment.get("commentType", "")).upper()
        document_type = type_mapping.get(comment_type)
        if document_type is None:
            continue
        for text_index, text_item in enumerate(comment.get("texts", [])):
            text = str(text_item.get("value", "")).strip()
            if not text:
                continue
            documents.append(
                KnowledgeDocument(
                    document_id=(f"uniprot:{accession}:{comment_type.lower()}:{comment_index}:{text_index}"),
                    text=text,
                    source="UniProt",
                    source_record_id=accession,
                    protein_id=accession,
                    gene_symbol=gene,
                    species_name=species,
                    taxonomy_id=taxonomy_id,
                    document_type=document_type,
                    section=comment_type.lower(),
                )
            )
    return documents


def _interpro_documents(records: list[dict[str, Any]], entry: dict[str, Any]) -> list[KnowledgeDocument]:
    accession, gene, species, taxonomy_id, _ = _identity(entry)
    documents = []
    for record in records:
        metadata = record.get("metadata", {})
        interpro_id = str(metadata.get("accession", "")).strip()
        name = str(metadata.get("name", "")).strip()
        if not interpro_id or not name:
            continue
        locations = []
        for protein in record.get("proteins", []):
            for location in protein.get("entry_protein_locations", []):
                for fragment in location.get("fragments", []):
                    start, end = fragment.get("start"), fragment.get("end")
                    if start is not None and end is not None:
                        locations.append(f"{start}-{end}")
        location_text = f"; residues {', '.join(locations)}" if locations else ""
        documents.append(
            KnowledgeDocument(
                document_id=f"interpro:{accession}:{interpro_id}",
                text=f"InterPro annotation {interpro_id}: {name}{location_text}.",
                source="InterPro",
                source_record_id=interpro_id,
                protein_id=accession,
                gene_symbol=gene,
                species_name=species,
                taxonomy_id=taxonomy_id,
                document_type="protein_domain",
                section="domain_annotation",
            )
        )
    return documents


async def _ingest_via_backend(
    base_url: str, documents: list[KnowledgeDocument], api_key: str | None
) -> tuple[int, int]:
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/v1/knowledge/protein/ingestions",
            headers=headers,
            json=[document.model_dump(mode="json") for document in documents],
        )
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise RuntimeError(body["error"].get("detail", "Knowledge ingestion failed"))
    return int(body["data"]["inserted"]), int(body["data"]["skipped"])


async def main(accession: str, base_url: str) -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        raise SystemExit("QDRANT_URL is required; refusing to seed an in-memory store")
    uniprot = UniProtClient(settings)
    interpro = InterProClient(settings)
    try:
        entry, annotations = await asyncio.gather(
            uniprot.get_entry(accession), interpro.annotations(accession)
        )
    finally:
        await uniprot.close()
        await interpro.close()
    documents = _uniprot_documents(entry) + _interpro_documents(annotations, entry)
    inserted, skipped = await _ingest_via_backend(base_url, documents, settings.internal_ingestion_api_key)
    print(
        f"Indexed {inserted} real document(s), skipped {skipped}, "
        f"protein={accession}, collection={settings.qdrant_collection}"
    )


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", default="P04637")
    parser.add_argument("--base-url", default=settings.local_base_url)
    args = parser.parse_args()
    asyncio.run(main(args.accession, args.base_url))
