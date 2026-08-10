"""Estimated cost is opt-in: unset pricing must never be guessed."""

from collections.abc import Iterator

import pytest

from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.domain.models import LlmUsage
from backend.agents.Protein_visualization.app.orchestrators.protein.orchestrator import _llm_usage

USAGE = LlmUsage(
    node="generate_grounded_explanation",
    model="gpt-4o",
    duration_ms=812,
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
)


@pytest.fixture(autouse=True)
def _fresh_settings_cache() -> Iterator[None]:
    """`get_settings` is `@lru_cache`d; each test needs its own env read."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cost_is_unset_without_configured_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_INPUT_PRICE_PER_1K_USD", raising=False)
    monkeypatch.delenv("AZURE_OUTPUT_PRICE_PER_1K_USD", raising=False)

    response = _llm_usage(USAGE)

    assert response.total_tokens == 1500
    assert response.estimated_cost_usd is None


def test_cost_is_computed_from_configured_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_1K_USD", "0.005")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_1K_USD", "0.015")

    response = _llm_usage(USAGE)

    # 1000 input tokens @ $0.005/1k + 500 output tokens @ $0.015/1k
    assert response.estimated_cost_usd == pytest.approx(0.005 + 0.0075)


def test_cost_requires_both_prices_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_1K_USD", "0.005")
    monkeypatch.delenv("AZURE_OUTPUT_PRICE_PER_1K_USD", raising=False)

    assert _llm_usage(USAGE).estimated_cost_usd is None


def test_cost_is_unset_when_token_counts_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_1K_USD", "0.005")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_1K_USD", "0.015")

    usage = LlmUsage(node="run_scientific_critic", model="gpt-4o", duration_ms=200)

    assert _llm_usage(usage).estimated_cost_usd is None
