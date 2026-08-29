from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
)

from .client import get_client, COLLECTION_NAME, VECTOR_SIZE




# Qdrant refuses to filter on a payload key that has no index, so any field
# used in a Filter has to be declared here. Constraint filtering in
# ingestion.retrieval depends on these.
PAYLOAD_INDEXES = (
    ("is_oa", PayloadSchemaType.BOOL),
    ("is_in_doaj", PayloadSchemaType.BOOL),
    ("apc_usd", PayloadSchemaType.INTEGER),
)


def ensure_payload_indexes():
    """Create the payload indexes needed for filtering. Safe to re-run."""

    created = []

    for field_name, schema in PAYLOAD_INDEXES:

        try:
            get_client().create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            created.append(field_name)

        except Exception as error:
            # Already-indexed is the common case on a re-run, not a failure.
            print(f"  index '{field_name}': {type(error).__name__} - {error}")

    return created


def create_journal_collection(recreate: bool = False):

    if get_client().collection_exists(COLLECTION_NAME):

        if not recreate:
            print(
                f"Collection '{COLLECTION_NAME}' already exists."
            )
            return False

        get_client().delete_collection(COLLECTION_NAME)

    get_client().create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    return True


if __name__ == "__main__":

    created = create_journal_collection()

    if created:
        print(f"'{COLLECTION_NAME}' collection created")

    print("Ensuring payload indexes...")

    indexed = ensure_payload_indexes()

    print(f"Payload indexes ready: {', '.join(indexed) or 'none added'}")
