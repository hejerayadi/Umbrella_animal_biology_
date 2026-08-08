from __future__ import annotations

import os
import pytest

from ..workflows.llm import get_llm_client


@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set — skipping live LLM smoke test",
)
def test_llm_client_round_trip():
    client = get_llm_client()
    response = client.invoke([{"role": "user", "content": "Say hello in one word."}])
    assert response.content
    assert len(response.content) > 0
