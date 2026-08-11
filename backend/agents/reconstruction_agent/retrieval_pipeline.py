from __future__ import annotations

from typing import Any

from .data_ingestion.embedding import k_mer_encode


class RetrievalPipeline:
    def __init__(self, qdrant_url: str, collection_name: str, top_k: int = 5):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.top_k = top_k
        self.client: Any | None = None

    def _get_client(self) -> Any | None:
        if self.client is not None:
            return self.client
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception:
            return None
        return QdrantClient(url=self.qdrant_url)

    def get_flanking_context(
        self,
        species_id: str,
        gap_start: int,
        gap_end: int,
        cleaned_sequence: str,
        species_metadata: dict | None = None,
    ) -> tuple[str, str]:
        """Return (left_context, right_context) of up to 1000 bp each.

        Extinction status is read exclusively from ``species_metadata["is_extinct"]``.
        Pass the full ``species_metadata`` dict produced by ``input_manager``.
        """
        is_extinct = bool((species_metadata or {}).get("is_extinct", False))
        left_flank = cleaned_sequence[max(0, gap_start - 1000):gap_start]
        right_flank = cleaned_sequence[gap_end:min(len(cleaned_sequence), gap_end + 1000)]

        if not cleaned_sequence:
            return left_flank, right_flank

        client = self._get_client()
        if client is None:
            return left_flank, right_flank

        query_vector = k_mer_encode(cleaned_sequence[max(0, gap_start - 50):gap_start + 50], k=4)
        try:
            raw_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
                filter={"must": [{"key": "species_id", "match": {"value": species_id}}]},
            )
        except Exception:
            return left_flank, right_flank

        results = []
        for item in raw_results or []:
            payload = item.get("payload", {}) if isinstance(item, dict) else {}
            if is_extinct is False and bool(payload.get("is_extinct", False)):
                continue
            results.append(item)

        if not results:
            return left_flank, right_flank

        left_parts: list[str] = []
        right_parts: list[str] = []
        for item in results:
            payload = item.get("payload", {}) if isinstance(item, dict) else {}
            window_sequence = payload.get("window_sequence", "")
            if not window_sequence:
                continue
            half = len(window_sequence) // 2
            left_parts.append(window_sequence[:half])
            right_parts.append(window_sequence[half:])

        if left_parts:
            left_flank = "".join(left_parts)[:1000]
        if right_parts:
            right_flank = "".join(reversed(right_parts))[:1000]
        return left_flank, right_flank
