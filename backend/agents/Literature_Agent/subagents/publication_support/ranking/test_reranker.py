from llm_reranker import rerank_journals


topic = "machine learning and artificial intelligence"


candidates = [
    {
        "name": "Neurocomputing",
        "publisher": "Elsevier BV",
        "semantic_score": 0.7718,
        "topic_score": 0.8826,
        "subfield_score": 0.8451,
        "field_score": 0.7409,
        "domain_score": 0.6173,
        "final_score": 0.8092,
    },
    {
        "name": "Expert Systems with Applications",
        "publisher": "Elsevier BV",
        "semantic_score": 0.7731,
        "topic_score": 0.8826,
        "subfield_score": 0.8451,
        "field_score": 0.7307,
        "domain_score": 0.6173,
        "final_score": 0.8091,
    },
    {
        "name": "Journal of the American Statistical Association",
        "publisher": "Taylor & Francis",
        "semantic_score": 0.7668,
        "topic_score": 0.8826,
        "subfield_score": 0.8451,
        "field_score": 0.7409,
        "domain_score": 0.6173,
        "final_score": 0.8070,
    },
    {
        "name": "Information Sciences",
        "publisher": "Elsevier BV",
        "semantic_score": 0.7618,
        "topic_score": 0.8826,
        "subfield_score": 0.8451,
        "field_score": 0.7409,
        "domain_score": 0.6173,
        "final_score": 0.8047,
    },
    {
        "name": "Nature",
        "publisher": "Nature Portfolio",
        "semantic_score": 0.7500,
        "topic_score": 0.8500,
        "subfield_score": 0.8200,
        "field_score": 0.7600,
        "domain_score": 0.6500,
        "final_score": 0.7900,
    },
]


result = rerank_journals(
    topic=topic,
    candidates=candidates,
    top_k=5,
)


print("\nLLM RE-RANKED RESULTS")
print("=" * 50)

for journal in result["ranked_journals"]:
    print(
        f"{journal['rank']}. "
        f"{journal['name']} "
        f"(score: {journal['score']})"
    )

    print(f"   {journal['reason']}")