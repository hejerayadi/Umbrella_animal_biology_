from pathlib import Path
import os

from backend.agents.reconstruction_agent.data_ingestion.embedding import k_mer_encode, EMBEDDING_DIM
from backend.agents.reconstruction_agent.data_ingestion.qc import clean_scaffolds
from backend.agents.reconstruction_agent.data_ingestion.windowing import extract_windows, mask_window
from backend.agents.reconstruction_agent.data_ingestion.storage import store_window_embedding


def test_clean_scaffolds_rejects_short_or_nheavy_sequences(tmp_path):
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">seq1\nA" * 1000, encoding="utf-8")

    cleaned = clean_scaffolds(fasta_path, "acc", "species")

    assert cleaned == []


def test_extract_windows_and_mask_window_have_expected_bounds(tmp_path):
    scaffold_sequence = "A" * 2500
    scaffold = type("Scaffold", (), {"sequence": scaffold_sequence, "scaffold_id": "scaf1"})()

    windows = extract_windows(type("CleanedScaffold", (), {"sequence": scaffold_sequence, "scaffold_id": "scaf1"})())
    assert len(windows) == 1

    masked = mask_window(windows[0], seed=42)
    assert 10 <= len(masked.masked_region) <= 500
    assert len(masked.left_context) + len(masked.masked_region) + len(masked.right_context) == 2000


def test_k_mer_encode_returns_normalized_vector():
    vector = k_mer_encode("ACGTACGT", k=4)
    assert len(vector) == EMBEDDING_DIM
    assert sum(v * v for v in vector) == 1.0


def test_store_window_embedding_logs_and_continues_on_failure():
    os.environ["TESTING"] = "true"
    class FakeQdrant:
        def __init__(self):
            self.calls = 0

        def upsert(self, collection_name, points):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")

    client = FakeQdrant()
    first_result = store_window_embedding(client, "collection", {"id": "p1", "vector": [0.1], "payload": {}})
    second_result = store_window_embedding(client, "collection", {"id": "p2", "vector": [0.1], "payload": {}})

    assert first_result is True
    assert second_result is False
