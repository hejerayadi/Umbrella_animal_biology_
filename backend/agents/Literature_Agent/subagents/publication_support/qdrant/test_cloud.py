"""Connection check: run this to confirm .env reaches Qdrant Cloud.

    python3 -m qdrant.test_cloud
"""

from qdrant.client import get_client, COLLECTION_NAME


client = get_client()

print("Connected.")
print("Collections:", client.get_collections())

if client.collection_exists(COLLECTION_NAME):

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    print(f"'{COLLECTION_NAME}' holds {count} points.")

else:
    print(
        f"'{COLLECTION_NAME}' does not exist yet — "
        f"run: python3 -m qdrant.qdrant_setup"
    )
