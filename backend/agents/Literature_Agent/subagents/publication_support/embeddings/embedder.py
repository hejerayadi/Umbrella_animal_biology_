from functools import lru_cache

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# Built on first use, never at import. This module sits on the import chain
# behind the Literature Agent's writing sub-orchestrator, so constructing the
# model here would make every `import` of the agent download and load ~130MB
# of weights - including on the routes that never recommend a journal.
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def build_journal_text(journal) -> str:

    parts = [
        f"Academic journal: {journal.name}"
    ]

    if journal.publisher:
        parts.append(
            f"Publisher: {journal.publisher}"
        )

    # Topics are the strongest research signal.
    topic_texts = []

    for topic in journal.topics:

        hierarchy = [
            topic.name,
            topic.subfield,
            topic.field,
            topic.domain,
        ]

        hierarchy = [
            value for value in hierarchy
            if value
        ]

        topic_texts.append(
            " → ".join(hierarchy)
        )

    if topic_texts:
        parts.append(
            "Research areas: " +
            "; ".join(topic_texts)
        )

    return ". ".join(parts)


def embed_journal(journal):

    text = build_journal_text(journal)

    vector = get_model().encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def build_query_text(query: str) -> str:

    return (
        "Academic research topic: "
        f"{query}. "
        "Relevant research fields, subfields, "
        "research topics, and academic journals."
    )


def embed_query(query: str):

    vector = get_model().encode(
        build_query_text(query),
        normalize_embeddings=True
    )

    return vector.tolist()