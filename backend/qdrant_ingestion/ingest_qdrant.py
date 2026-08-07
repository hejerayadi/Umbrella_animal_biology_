"""
Chahd's Sprint 2 Qdrant ingestion pipeline.

Creates the real Sprint 2 collection and inserts a small set of deterministic
mock-vector reference points with the payload schema agreed with Leith
(see the Sprint 2 spec, section 7.3/7.4). Also writes manifest.json, which
must be shared with Leith so his query-side adapter matches this contract
exactly.

Install:
    pip install qdrant-client

Before running:
    1. Start Qdrant locally, e.g.: docker run -p 6333:6333 qdrant/qdrant
    2. Put a few demo images under demo_images/ (or edit DEMO_RECORDS below)
    3. Confirm COLLECTION_NAME / VECTOR_DIMENSION / DISTANCE with Leith

Run:
    python ingest_qdrant.py
"""

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from mock_bioclip2 import (
    embed_image_bytes,
    VECTOR_DIMENSION,
    MOCK_PROVIDER_VERSION,
    EMBEDDING_MODE,
)

# Loads QDRANT_URL and QDRANT_API_KEY from the .env file sitting next to this
# script. Never hardcode these values directly in this file.
load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# ---------------------------------------------------------------------------
# Sprint 2 manifest values — DEFAULTS, not yet frozen with Leith.
# Confirm these together, then treat them as fixed for the sprint.
# ---------------------------------------------------------------------------
COLLECTION_NAME = "sprint2_recognition_reference"
VECTOR_NAME = None  # unnamed vector; set a string key here if you want a named vector instead
DISTANCE = Distance.COSINE
DATASET_VERSION = "sprint2-qdrant-minimal-v1"

MANIFEST_PATH = Path("manifest.json")

# ---------------------------------------------------------------------------
# Demo reference set — replace image_path with your team's actual demo images.
# Each record becomes one Qdrant point.
# ---------------------------------------------------------------------------
DEMO_RECORDS = [
    {
        "reference_id": "sprint2-ref-001",
        "species_id": "panthera_leo",
        "scientific_name": "Panthera leo",
        "common_name": "lion",
        "image_path": "demo_images/lion_01.jpg",
    },
    {
        "reference_id": "sprint2-ref-002",
        "species_id": "panthera_tigris",
        "scientific_name": "Panthera tigris",
        "common_name": "tiger",
        "image_path": "demo_images/tiger_01.jpg",
    },
    {
        "reference_id": "sprint2-ref-003",
        "species_id": "vulpes_vulpes",
        "scientific_name": "Vulpes vulpes",
        "common_name": "red fox",
        "image_path": "demo_images/fox_01.jpg",
    },
]


def create_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE),
    )
    print(
        f"[Qdrant] Created collection '{COLLECTION_NAME}' "
        f"(dim={VECTOR_DIMENSION}, distance={DISTANCE.value})."
    )


def build_points() -> list[PointStruct]:
    points = []
    for record in DEMO_RECORDS:
        image_path = Path(record["image_path"])
        if not image_path.exists():
            print(
                f"[WARN] Missing image {image_path}, skipping '{record['reference_id']}'. "
                f"Add the file or update DEMO_RECORDS."
            )
            continue

        vector = embed_image_bytes(image_path.read_bytes())

        payload = {
            "reference_id": record["reference_id"],
            "species_id": record["species_id"],
            "scientific_name": record["scientific_name"],
            "common_name": record["common_name"],
            "source": "Sprint 2 controlled fixture",
            "dataset_version": DATASET_VERSION,
            "embedding_mode": EMBEDDING_MODE,
            "mock_provider_version": MOCK_PROVIDER_VERSION,
        }

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    return points


def write_manifest() -> None:
    manifest = {
        "collection_name": COLLECTION_NAME,
        "vector_name": VECTOR_NAME,
        "vector_dimension": VECTOR_DIMENSION,
        "distance": DISTANCE.value,
        "embedding_mode": EMBEDDING_MODE,
        "mock_provider_version": MOCK_PROVIDER_VERSION,
        "dataset_version": DATASET_VERSION,
        "required_payload_fields": [
            "species_id",
            "scientific_name",
            "dataset_version",
            "embedding_mode",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[Manifest] Written to {MANIFEST_PATH.resolve()} — send this file to Leith.")


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    create_collection(client)

    points = build_points()
    if not points:
        print("[ERROR] No points built — add demo images under demo_images/ and rerun.")
        return

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Qdrant] Inserted {len(points)} point(s) into '{COLLECTION_NAME}'.")

    write_manifest()


if __name__ == "__main__":
    main()
