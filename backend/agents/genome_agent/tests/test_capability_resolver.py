from __future__ import annotations

import pytest

from ..workflows.agent_catalog import _get_known_agent_names
from ..workflows.capability_resolver import (
    _is_valid_target,
    resolve_capability,
    resolve_capability_fallback,
)


def test_resolve_capability_fallback_returns_known_agent():
    decision = resolve_capability_fallback(
        current_agent="visualization",
        prompt_to_target_agent="Render an interactive, labeled 3D structure for the requested gene.",
    )
    assert decision.target_agent != "none"
    assert _is_valid_target(decision.target_agent)


def test_resolve_capability_fallback_unknown_returns_none():
    decision = resolve_capability_fallback(
        current_agent="visualization",
        prompt_to_target_agent="Do something completely unknown and impossible.",
    )
    assert decision.target_agent == "none"


def test_is_valid_target_rejects_unknown():
    assert _is_valid_target("TotallyFakeAgent") is False


def test_is_valid_target_accepts_known():
    known = _get_known_agent_names()
    if known:
        assert _is_valid_target(known[0]) is True


@pytest.mark.skipif(
    not pytest.importorskip("os").getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set — skipping live LLM capability resolver test",
)
def test_resolve_capability_never_returns_absent_agent():
    known = _get_known_agent_names()
    for _ in range(3):
        decision = resolve_capability(
            current_agent="visualization",
            prompt_to_target_agent="Render an interactive, labeled 3D structure for the requested gene.",
            known_context={},
        )
        if decision is not None:
            assert decision.target_agent in known or decision.target_agent == "none"
