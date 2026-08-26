from models.journal import Journal, JournalTopic


def normalize_journal(data: dict) -> Journal:

    normalized_topics = []

    for topic in data.get("topics", []):

        topic_name = topic.get("display_name")

        if not topic_name:
            continue

        subfield = topic.get("subfield") or {}
        field = topic.get("field") or {}
        domain = topic.get("domain") or {}

        normalized_topics.append(
            JournalTopic(
                name=topic_name,
                subfield=subfield.get("display_name"),
                field=field.get("display_name"),
                domain=domain.get("display_name"),
            )
        )

    return Journal(
        id=data.get("id"),
        name=data.get("display_name"),
        publisher=data.get("host_organization_name"),
        issn=data.get("issn_l"),
        works_count=data.get("works_count", 0),
        cited_by_count=data.get("cited_by_count", 0),
        country=data.get("country_code"),
        homepage=data.get("homepage_url"),

        is_oa=bool(data.get("is_oa", False)),
        is_in_doaj=bool(data.get("is_in_doaj", False)),
        apc_usd=data.get("apc_usd"),

        topics=normalized_topics,
    )