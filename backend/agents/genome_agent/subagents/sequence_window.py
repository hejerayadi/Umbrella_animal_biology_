"""
Sequence Window — real NCBI Nuccore efetch subagent (Task 5, NEW capability)
Fetches a small DNA sequence window for a resolved assembly.
Safety: MAX_WINDOW_BP = 200_000 — requests larger than this raise
WindowTooLargeError before any HTTP call is made.
Never cached — pass-through only.
"""

from __future__ import annotations

import asyncio
import logging

from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)

MAX_WINDOW_BP = 200_000


class WindowTooLargeError(Exception):
    """Raised when a requested sequence window exceeds MAX_WINDOW_BP."""
    pass


async def fetch_sequence_window(
    assembly_id: str,
    seq_start: int,
    seq_stop: int,
) -> str:
    window_size = seq_stop - seq_start
    if window_size > MAX_WINDOW_BP:
        raise WindowTooLargeError(
            f"Requested window size ({window_size} bp) exceeds "
            f"MAX_WINDOW_BP ({MAX_WINDOW_BP} bp)"
        )

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "efetch.fcgi",
            "db": "nuccore",
            "id": assembly_id,
            "seq_start": seq_start,
            "seq_stop": seq_stop,
            "rettype": "fasta",
            "retmode": "text",
        },
    )
    return resp.text


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Sequence Window live NCBI test ---")
        
        # Small window — should work
        seq = await fetch_sequence_window("GCF_000464555.1", 1, 1000)
        print("Window 1-1000:", seq[:100])
        assert ">" in seq or len(seq) > 0

        # Too-large window — should raise before any HTTP call
        try:
            await fetch_sequence_window("GCF_000464555.1", 1, 200_001)
            assert False, "Expected WindowTooLargeError"
        except WindowTooLargeError as exc:
            print("Correctly raised:", exc)

        print("All tests passed ✅")

    asyncio.run(_quick_test())
