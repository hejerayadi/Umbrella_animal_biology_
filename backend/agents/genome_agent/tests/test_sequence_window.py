from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from ..subagents.sequence_window import WindowTooLargeError, fetch_sequence_window, MAX_WINDOW_BP


def test_window_too_large_raises_without_http_call():
    """Confirm MAX_WINDOW_BP guard raises before any network request."""
    with patch("backend.agents.genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        with pytest.raises(WindowTooLargeError):
            asyncio.run(
                fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP + 1)
            )
        mock_get.assert_not_called()


def test_window_at_max_size_plus_one_raises():
    """Confirm exactly MAX_WINDOW_BP + 1 raises."""
    with patch("backend.agents.genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        with pytest.raises(WindowTooLargeError):
            asyncio.run(
                fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP + 1)
            )
        mock_get.assert_not_called()


def test_window_just_under_max_succeeds():
    """Confirm exactly MAX_WINDOW_BP does NOT raise and does call HTTP."""
    with patch("backend.agents.genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.raise_for_status = lambda: None
        mock_response.text = ">test\nACGT"

        result = asyncio.run(
            fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP)
        )
        assert result == ">test\nACGT"
        mock_get.assert_called_once()
