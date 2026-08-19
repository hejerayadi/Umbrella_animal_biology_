"""
Ingestion du dataset arXiv (ccdv/arxiv-summarization) dans la collection
Qdrant ghaya_papers_fulltext, filtree strictement sur la biodiversite
ANIMALE (exclut plantes/algues/microbes -- pas de "biodiversity"/"species
diversity"/"ecosystem" generiques, uniquement des termes qui impliquent un
animal explicitement). Filtre applique sur l'abstract uniquement (signal
plus fiable que le debut du corps).

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.subagents.writing.kb.ingest_arxiv
"""

import re
import tiktoken
from datasets import load_dataset

from .ingest_to_qdrant import upsert_batch

MAX_TOKENS_PER_CHUNK = 200
COLLECTION = "ghaya_papers_fulltext"
DOMAIN = "biodiversity"

# Uniquement des termes qui impliquent un ANIMAL explicitement -- pas de mot
# generique type "biodiversity"/"species"/"ecosystem"/"phylogenetic" qui
# s'applique aussi bien aux plantes, algues, ou microbes.
ANIMAL_KEYWORDS = [
    "wildlife", "fauna", "zoology", "wild animal",
    "wildlife conservation", "species conservation", "conservation biology",
    "animal diversity", "faunal diversity",
    "mammal species", "avian species", "vertebrate species", "invertebrate species",
    "bird species", "fish species", "insect species", "reptile species", "amphibian species",
    "migratory bird", "animal migration", "predator species", "prey species",
    "endangered animal", "endangered wildlife",
]

_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ANIMAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def is_relevant(abstract: str, body: str) -> bool:
    # Filtre sur l'abstract uniquement -- signal plus fiable que le debut
    # du corps, qui digresse souvent vers du contexte hors-sujet.
    return bool(_KEYWORD_PATTERN.search(abstract))


def chunk_section_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(". ".join(current) + ".")
            current, current_tokens = [], 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(". ".join(current) + ".")

    return chunks


def build_points_for_article(article: dict, article_idx: int) -> tuple[list[str], list[dict], list[str]]:
    texts, payloads, ids = [], [], []

    article_id = str(article.get("id", article_idx))
    abstract = article.get("abstract", "") or ""
    body = article.get("article", "") or ""

    if abstract.strip():
        for chunk_idx, chunk in enumerate(chunk_section_text(abstract)):
            texts.append(chunk)
            payloads.append({
                "doc_id": article_id,
                "source_dataset": "arxiv",
                "domain": DOMAIN,
                "section_type": "abstract",
                "input_text": chunk,
                "target_text": abstract,
                "n_tokens": count_tokens(chunk),
                "language": "en",
            })
            ids.append(f"arxiv_{article_id}_abstract_{chunk_idx}")

    if body.strip():
        for chunk_idx, chunk in enumerate(chunk_section_text(body)):
            texts.append(chunk)
            payloads.append({
                "doc_id": article_id,
                "source_dataset": "arxiv",
                "domain": DOMAIN,
                "section_type": "body",
                "input_text": chunk,
                "target_text": abstract,
                "n_tokens": count_tokens(chunk),
                "language": "en",
            })
            ids.append(f"arxiv_{article_id}_body_{chunk_idx}")

    return texts, payloads, ids


def ingest_arxiv(target_matches: int = 100, max_scan: int = 40000):
    print("Chargement du dataset arXiv (ccdv/arxiv-summarization)...", flush=True)
    dataset = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)

    all_texts, all_payloads, all_ids = [], [], []
    scanned = 0
    matched = 0

    print(f"Scan en cours (cible : {target_matches} articles animaux stricts, max {max_scan} scannes)...", flush=True)
    for article in dataset:
        scanned += 1
        abstract = article.get("abstract", "") or ""
        body = article.get("article", "") or ""

        if is_relevant(abstract, body):
            texts, payloads, ids = build_points_for_article(article, scanned)
            all_texts.extend(texts)
            all_payloads.extend(payloads)
            all_ids.extend(ids)
            matched += 1

        if scanned % 500 == 0:
            print(f"  ... {scanned} scannes, {matched} retenus jusqu'ici", flush=True)

        if matched >= target_matches or scanned >= max_scan:
            break

    print(f"{scanned} articles scannes -> {matched} retenus -> {len(all_texts)} chunks a inserer dans {COLLECTION}", flush=True)
    if all_texts:
        upsert_batch(COLLECTION, all_texts, all_payloads, all_ids, batch_size=20)
    print("Ingestion arXiv terminee.", flush=True)


if __name__ == "__main__":
    # arXiv est generaliste (physique/maths/CS majoritaires) : les papiers
    # strictement animaux y sont rares, on vise moins d'articles mais on
    # scanne plus large pour compenser.
    ingest_arxiv(target_matches=30, max_scan=50000)