"""Shared test configuration.

Tests that reach a real external service are marked `external` and skipped
unless explicitly enabled. The default suite must run offline and fast: a
suite that silently depends on EMBL-EBI being responsive is a suite that fails
for reasons unrelated to the change under test.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("RUN_EXTERNAL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="Set RUN_EXTERNAL_TESTS=1 to run tests that hit real services.")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)
