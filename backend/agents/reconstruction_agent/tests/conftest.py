"""Shared fixtures.

The whole suite runs offline. Anything that would touch NCBI or EMBL-EBI is
marked `external` and skipped unless RUN_EXTERNAL_TESTS=1, so a normal `pytest`
run is fast, deterministic, and safe to run repeatedly without burning through
a published rate limit.
"""
from __future__ import annotations

import os

import pytest

from configuration.runtime import use_selector_event_loop
from configuration.settings import (
    EMBLEBISettings,
    LLMSettings,
    NCBISettings,
    Settings,
)
from domain.models import Reference, Sequence
from observability.events import CollectingEmitter

# Must run before pytest-asyncio creates a loop, or any test touching psycopg
# fails on Windows with a ProactorEventLoop error. Import-time on purpose:
# a fixture would run too late, after the loop already exists.
use_selector_event_loop()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `external` tests unless explicitly enabled."""
    if os.getenv("RUN_EXTERNAL_TESTS") == "1":
        return

    skip = pytest.mark.skip(reason="Set RUN_EXTERNAL_TESTS=1 to run tests that hit real services.")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings() -> Settings:
    """Settings with no external credentials and no LLM.

    Constructed directly rather than through `get_settings()` so a developer's
    real `.env` cannot change what the tests assert.
    """
    return Settings(
        max_iterations=3,
        min_confidence=0.55,
        max_gap_length=5000,
        ncbi=NCBISettings(contact_email="tests@umbrella.local"),
        embl_ebi=EMBLEBISettings(contact_email="tests@umbrella.local"),
        llm=LLMSettings(),
    )


@pytest.fixture
def events() -> CollectingEmitter:
    return CollectingEmitter()


@pytest.fixture
def gapped_sequence() -> Sequence:
    """A sequence with two gaps: a 10-base run and a 5-base run.

    Flanks are long enough to be realistic but short enough to read in a
    failure message.
    """
    left = "ACGT" * 20  # 80 bases
    middle = "TTAC" * 20  # 80 bases
    right = "GGCA" * 20  # 80 bases
    return Sequence.parse(
        "test_seq",
        f"{left}{'N' * 10}{middle}{'N' * 5}{right}",
        organism="Testus organismus",
    )


@pytest.fixture
def ungapped_sequence() -> Sequence:
    return Sequence.parse("complete_seq", "ACGT" * 50, organism="Testus organismus")


@pytest.fixture
def references() -> list[Reference]:
    """Three references spanning the useful range of quality."""
    return [
        Reference(
            accession="REF_STRONG",
            organism="Testus organismus",
            residues="ACGT" * 40,
            identity=0.98,
            coverage=0.95,
            relatedness=1.0,
            source="blast",
        ),
        Reference(
            accession="REF_MEDIUM",
            organism="Testus relatus",
            residues="ACGT" * 40,
            identity=0.85,
            coverage=0.80,
            relatedness=0.8,
            source="blast",
        ),
        Reference(
            accession="REF_WEAK",
            organism="Distantus species",
            residues="TGCA" * 40,
            identity=0.55,
            coverage=0.40,
            relatedness=0.3,
            source="blast",
        ),
    ]
