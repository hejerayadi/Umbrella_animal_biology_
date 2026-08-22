"""Pytest configuration — ensure repository root is on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _disable_live_protein_evidence(monkeypatch):
    """Existing routing/API tests must not call UniProt, PDB, or Serper."""
    from backend.agents.image_generation_agent.tool_schemas import EvidenceBundle

    def _skip_live_evidence(**kwargs):
        return EvidenceBundle.empty()

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.gather_protein_evidence",
        _skip_live_evidence,
    )

