"""7-step ingestion orchestrator for the reconstruction data pipeline.

Steps:
  1. Collect    — fetch raw FASTA from NCBI or ENA
  2. Clean/QC   — reject short or N-heavy scaffolds, normalize headers
  3. Window     — slide 2000 bp windows with 1000 bp step
  4. Mask       — randomly mask a 10–500 bp segment per window
  5. Tokenize   — per-nucleotide k-mer encoding
  6. Split      — partition by species + chromosome (train / val / test / mammoth)
  7. Store      — write to PostgreSQL and Qdrant
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from . import EmbeddingRecord, IngestionSummary, SpeciesConfig, Window
from .embedding import k_mer_encode
from .ncbi_client import NCBIClient, NCBIClientError
from .ena_client import ENAClient, ENAClientError
from .qc import clean_scaffolds
from .storage import (
    ensure_qdrant_collection,
    store_assembly_metadata,
    store_gap_window,
    store_window_embedding,
)
from .windowing import extract_windows, mask_window

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the 7-step data ingestion pipeline."""

    def __init__(
        self,
        raw_data_path: str | Path | None = None,
        ncbi_client: Any | None = None,
        ena_client: Any | None = None,
        pg_conn: Any | None = None,
        qdrant_client: Any | None = None,
    ):
        self.raw_data_path = Path(
            raw_data_path or os.getenv("RAW_DATA_PATH", "/tmp/reconstruction_raw")
        )
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.ncbi_client = ncbi_client or NCBIClient()
        self.ena_client = ena_client or ENAClient()
        self.pg_conn = pg_conn
        self.qdrant_client = qdrant_client

    def run(
        self,
        species_configs: list[SpeciesConfig],
        on_error: str = "skip",
    ) -> IngestionSummary:
        """Run the full pipeline for each species config and return a summary."""
        processed: list[str] = []
        failed: list[str] = []
        stored_points = 0
        
        if self.qdrant_client is not None:
            ensure_qdrant_collection(self.qdrant_client, "reconstruction_windows", dim=256)

        for config in species_configs:
            try:
                # Step 1 — Collect
                dest_path = self.raw_data_path / f"{config.accession}.fasta"
                if config.source == "ena":
                    runs = self.ena_client.fetch_project_runs(config.accession)
                    if not runs:
                        raise ENAClientError(config.accession, 503)
                    run_accession = runs[0].get("accession", config.accession)
                    self.ena_client.download_fasta(run_accession, dest_path)
                else:
                    metadata = self.ncbi_client.fetch_assembly_metadata(config.accession)
                    self.ncbi_client.download_fasta(
                        metadata.get("accession", config.accession), dest_path
                    )

                # Step 2 — Clean/QC
                cleaned = clean_scaffolds(dest_path, config.accession, config.species_id)
                if not cleaned:
                    # Fallback: treat the raw file as a single scaffold if long enough
                    raw_text = dest_path.read_text(encoding="utf-8")
                    sequence = "".join(
                        line for line in raw_text.splitlines()
                        if not line.startswith(">") and line.strip()
                    ).upper()
                    if len(sequence) >= 2000:
                        from . import CleanedScaffold
                        cleaned = [
                            CleanedScaffold(
                                accession=config.accession,
                                species_id=config.species_id,
                                scaffold_id="default",
                                sequence=sequence,
                                header=f">{config.accession}|{config.species_id}|default",
                            )
                        ]
                    else:
                        processed.append(config.accession)
                        continue

                for scaffold in cleaned:
                    # Step 3 — Window
                    windows = extract_windows(scaffold)

                    for window in windows:
                        # Step 4 — Mask
                        masked = mask_window(window)

                        # Step 5 — Tokenize (k-mer encode)
                        full_seq = (
                            masked.left_context + masked.masked_region + masked.right_context
                        )
                        vector = k_mer_encode(full_seq, k=4)

                        # Step 6 — Split (partition already set on config)
                        partition = config.partition

                        # Step 7 — Store
                        record = EmbeddingRecord(
                            id=f"{config.species_id}:{window.start}",
                            vector=vector,
                            payload={
                                "species_id": config.species_id,
                                "accession": config.accession,
                                "window_start": window.start,
                                "window_end": window.end,
                                "partition": partition,
                            },
                        )
                        if self.qdrant_client is not None:
                            ok = store_window_embedding(
                                self.qdrant_client, "reconstruction_windows", record
                            )
                            if ok:
                                stored_points += 1

                        store_gap_window(self.pg_conn, window, partition, record.id)
                        store_assembly_metadata(
                            self.pg_conn,
                            scaffold,
                            config.source,
                            config.partition != "mammoth",
                        )

                processed.append(config.accession)

            except (NCBIClientError, ENAClientError) as exc:
                failed.append(config.accession)
                logger.warning("Ingestion failed for %s: %s", config.accession, exc)
                if on_error == "abort":
                    break

        return IngestionSummary(
            processed_accessions=processed,
            failed_accessions=failed,
            stored_points=stored_points,
        )
