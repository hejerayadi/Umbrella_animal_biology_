"""Connection check: run this to confirm .env reaches Qdrant Cloud.

    python3 -m qdrant.test_cloud
"""

from .client import get_client, COLLECTION_NAME



print("Connected.")
print("Collections:", get_client().get_collections())

if get_client().collection_exists(COLLECTION_NAME):

    count = get_client().count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    print(f"'{COLLECTION_NAME}' holds {count} points.")

else:
    print(
        f"'{COLLECTION_NAME}' does not exist yet — "
        f"run: python3 -m qdrant.qdrant_setup"
    )
