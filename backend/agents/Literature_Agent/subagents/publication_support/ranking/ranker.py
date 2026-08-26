import numpy as np

from embeddings.embedder import model


# Topic labels repeat heavily across journals (OpenAlex has a few thousand
# topics, ~250 subfields, 26 fields and 4 domains), so caching each label's
# vector turns thousands of encodes per query into a few hundred.
_embedding_cache = {}


# The keys used inside a payload topic dict, and the score name each maps to.
HIERARCHY_LEVELS = (
    ("name", "topic"),
    ("subfield", "subfield"),
    ("field", "field"),
    ("domain", "domain"),
)


def embed_texts(texts):
    """Embed many texts at once, reusing anything already seen."""

    missing = [
        text for text in dict.fromkeys(texts)
        if text not in _embedding_cache
    ]

    if missing:

        vectors = model.encode(
            missing,
            batch_size=64,
            normalize_embeddings=True
        )

        for text, vector in zip(missing, vectors):
            _embedding_cache[text] = vector

    return [_embedding_cache[text] for text in texts]


def warm_cache(texts):
    """Pre-embed labels in one batch before scoring candidates."""

    if texts:
        embed_texts(texts)


def cosine_similarity(a, b):

    # Vectors are already L2-normalised, so the dot product is the cosine.
    return float(np.dot(a, b))


def text_similarity(text1: str, text2: str):

    vector1, vector2 = embed_texts([text1, text2])

    return cosine_similarity(vector1, vector2)


def topic_relevance(query: str, topics: list):

    names = [
        topic.get("name")
        for topic in topics
        if topic.get("name")
    ]

    if not names:
        return 0.0

    query_vector = embed_texts([query])[0]

    scores = [
        cosine_similarity(query_vector, vector)
        for vector in embed_texts(names)
    ]

    # The best matching topic is the strongest signal.
    return max(scores)


def hierarchy_relevance(query: str, topics: list):

    scores = {
        score_name: 0.0
        for _, score_name in HIERARCHY_LEVELS
    }

    if not topics:
        return scores

    query_vector = embed_texts([query])[0]

    for topic_key, score_name in HIERARCHY_LEVELS:

        labels = [
            topic.get(topic_key)
            for topic in topics
            if topic.get(topic_key)
        ]

        if not labels:
            continue

        scores[score_name] = max(
            cosine_similarity(query_vector, vector)
            for vector in embed_texts(labels)
        )

    return scores


def calculate_final_score(
    semantic_score: float,
    hierarchy_scores: dict
):

    topic_score = hierarchy_scores["topic"]
    subfield_score = hierarchy_scores["subfield"]
    field_score = hierarchy_scores["field"]
    domain_score = hierarchy_scores["domain"]

    final_score = (
        0.45 * semantic_score
        + 0.30 * topic_score
        + 0.15 * subfield_score
        + 0.07 * field_score
        + 0.03 * domain_score
    )

    return final_score
