# Qdrant setup - species_taxonomy collection

The Biodiversity Orchestrator uses Qdrant to normalize free-form species
names to their scientific counterparts before dispatching to workers.
This note walks through the one-time bootstrap.

## 1. Credentials

Generate an API key from the Qdrant Cloud dashboard for the
``umbrella_project`` cluster ("Create an API Key" on the cluster card)
and copy the cluster URL. Put both in your local ``.env``:

```
QDRANT_URL=https://<cluster-id>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<key>
```

Never commit these values - the ``.gitignore`` in this package already
excludes ``.env``.

## 2. Collection layout

- Collection name: ``species_taxonomy``
- Vector size: 384 (matches ``sentence-transformers/all-MiniLM-L6-v2``)
- Distance metric: Cosine
- Payload (per point):
    - ``scientific_name``: canonical binomial, e.g. ``Loxodonta africana``
    - ``common_names``: list of common / vernacular names in any language
    - ``taxonomic_rank``: ``species`` today (kept for future genus points)

## 3. Bootstrap script (draft)

Not required for Sprint 2 acceptance - the orchestrator falls back to a
dict when Qdrant is unreachable. When we are ready to run against the
real cluster, the ingestion script will look roughly like:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client.recreate_collection(
    collection_name="species_taxonomy",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
points = []
for i, (sci, commons) in enumerate(CATALOGUE):
    text = sci + " | " + " | ".join(commons)
    points.append(
        PointStruct(
            id=i,
            vector=embedder.encode(text, normalize_embeddings=True).tolist(),
            payload={"scientific_name": sci, "common_names": commons, "taxonomic_rank": "species"},
        )
    )
client.upsert(collection_name="species_taxonomy", points=points)
```

The catalogue itself will be a curated list of the ~10k mammal /
bird / reptile species that show up most often in GBIF, seeded from
IUCN's own name list.

## 4. Namespacing (multi-group concern)

The ``umbrella_project`` cluster is shared across every group. If Hajer
asks us to namespace, prefix the collection name with the group
identifier - ``group_e_species_taxonomy`` - and update ``COLLECTION_NAME``
in ``services/qdrant_client.py`` accordingly.
