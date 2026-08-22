from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ..subagents.sequence_window import (
    MAX_WINDOW_BP,
    NoLinkedSequenceError,
    WindowTooLargeError,
    fetch_sequence_window,
)


def test_window_too_large_raises_without_http_call():
    """Confirm MAX_WINDOW_BP guard raises before any network request."""
    with patch("genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        with pytest.raises(WindowTooLargeError):
            asyncio.run(
                fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP + 1)
            )
        mock_get.assert_not_called()


def test_window_at_max_size_plus_one_raises():
    """Confirm exactly MAX_WINDOW_BP + 1 raises, still with no HTTP call."""
    with patch("genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        with pytest.raises(WindowTooLargeError):
            asyncio.run(
                fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP + 1)
            )
        mock_get.assert_not_called()


def _mock_response(*, json_data: dict | None = None, text: str = "", status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = lambda: None
    resp.json = lambda: json_data or {}
    resp.text = text
    return resp


def test_window_just_under_max_resolves_and_fetches():
    """Confirm the full esearch -> elink -> efetch chain runs and returns
    sequence text when everything resolves. This is the regression test for
    the original bug: passing the assembly accession straight to efetch on
    db=nuccore, which NCBI rejects with "Failed to understand id"."""
    esearch_resp = _mock_response(json_data={"esearchresult": {"idlist": ["999"]}})
    elink_resp = _mock_response(
        json_data={
            "linksets": [
                {
                    "dbfrom": "assembly",
                    "linksetdbs": [
                        {"dbto": "nuccore", "linkname": "assembly_nuccore_refseq", "links": ["12345"]}
                    ],
                }
            ]
        }
    )
    efetch_resp = _mock_response(text=">test\nACGT")

    with patch("genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        mock_get.side_effect = [esearch_resp, elink_resp, efetch_resp]

        result = asyncio.run(
            fetch_sequence_window("GCF_000464555.1", 0, MAX_WINDOW_BP)
        )

        assert result == ">test\nACGT"
        assert mock_get.call_count == 3

        # The final call (efetch) must use the *nuccore* UID from elink
        # ("12345"), never the assembly accession itself.
        efetch_call_params = mock_get.call_args_list[-1].kwargs["params"]
        assert efetch_call_params["id"] == "12345"


def test_assembly_not_found_raises_no_linked_sequence_error():
    """If the assembly accession itself doesn't resolve to a UID, fail
    clearly instead of attempting elink/efetch with nothing to link from."""
    esearch_resp = _mock_response(json_data={"esearchresult": {"idlist": []}})

    with patch("genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        mock_get.side_effect = [esearch_resp]

        with pytest.raises(NoLinkedSequenceError):
            asyncio.run(fetch_sequence_window("GCF_not_real.1", 0, 100))

        assert mock_get.call_count == 1


def test_no_nuccore_link_raises_no_linked_sequence_error():
    """If the assembly resolves but has no linked Nuccore sequence under
    either linkname, fail clearly instead of calling efetch with an empty ID."""
    esearch_resp = _mock_response(json_data={"esearchresult": {"idlist": ["999"]}})
    empty_elink_resp = _mock_response(json_data={"linksets": [{"linksetdbs": []}]})

    with patch("genome_agent.subagents._ncbi_client.requests.get") as mock_get:
        # Both linkname attempts (refseq, then insdc) come back empty.
        mock_get.side_effect = [esearch_resp, empty_elink_resp, empty_elink_resp]

        with pytest.raises(NoLinkedSequenceError):
            asyncio.run(fetch_sequence_window("GCF_000464555.1", 0, 100))

        assert mock_get.call_count == 3
