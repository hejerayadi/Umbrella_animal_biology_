from qdrant_client.models import PointStruct

from ..embeddings.embedder import embed_journal
from .normalizer import normalize_journal
from ..qdrant.client import get_client, COLLECTION_NAME

import json
import time



DATA_PATH = "data/journals.json"

# One upsert of every point is fine over a local socket, but not over the
# network: each journal carries up to 25 topics, so 256 points is a large
# enough request to time out against the cloud.
UPSERT_BATCH_SIZE = 64

UPSERT_RETRIES = 3


def build_points(data):

    points = []

    for i, journal_data in enumerate(data):

        journal = normalize_journal(journal_data)

        vector = embed_journal(journal)

        topics = [
            {
                "name": topic.name,
                "subfield": topic.subfield,
                "field": topic.field,
                "domain": topic.domain,
            }
            for topic in journal.topics
        ]

        payload = {
            "id": journal.id,
            "name": journal.name,
            "publisher": journal.publisher,
            "issn": journal.issn,

            "works_count": journal.works_count,
            "cited_by_count": journal.cited_by_count,

            "country": journal.country,
            "homepage": journal.homepage,

            "is_oa": journal.is_oa,
            "is_in_doaj": journal.is_in_doaj,
            "apc_usd": journal.apc_usd,

            "topics": topics,
        }

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload=payload
            )
        )

    return points


def upsert_batch(batch):
    """Upsert one batch, retrying on network timeouts.

    Point ids are deterministic, so retrying a batch that partially
    landed simply overwrites the same points.
    """

    for attempt in range(1, UPSERT_RETRIES + 1):

        try:
            get_client().upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True
            )
            return

        except Exception as error:

            if attempt == UPSERT_RETRIES:
                raise

            print(
                f"  retry {attempt}/{UPSERT_RETRIES - 1} "
                f"after: {type(error).__name__}"
            )

            time.sleep(2 * attempt)


def ingest(path: str = DATA_PATH):

    with open(path, "r") as f:
        data = json.load(f)

    points = build_points(data)

    for start in range(0, len(points), UPSERT_BATCH_SIZE):

        batch = points[start:start + UPSERT_BATCH_SIZE]

        upsert_batch(batch)

        print(
            f"Upserted {start + len(batch)}/{len(points)}"
        )

    return points


if __name__ == "__main__":

    points = ingest()

    print(
        f"Inserted {len(points)} journals into Qdrant."
    )
