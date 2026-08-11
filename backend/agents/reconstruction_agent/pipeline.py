"""Backward-compatibility shim — IngestionPipeline now lives in data_ingestion/pipeline.py."""

from .data_ingestion.pipeline import IngestionPipeline  # noqa: F401

__all__ = ["IngestionPipeline"]
