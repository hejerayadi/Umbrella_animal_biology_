"""
Live NCBI check for the Genome Agent's real subagents.

Unlike the pytest suite (which keeps the NCBI-hitting tests opt-in and
skipped by default, see tests/test_ncbi_smoke.py), this is a plain
executable script meant to be run by hand whenever you want to eyeball
what NCBI is actually returning right now. It calls every real subagent
added in the NCBI integration directly — no mocking, no fallback paths:

    1. species_resolver.resolve_species()      — NCBI Assembly esearch/esummary
    2. genome_metadata.get_genome_metadata()    — NCBI Assembly esummary
    3. gene_annotation.get_gene_annotation()    — NCBI Gene esearch/esummary
    4. sequence_window.fetch_sequence_window()  — NCBI Nuccore efetch
    5. visualization.generate_visualization()   — size_comparison, which
       itself calls resolve_species()/get_genome_metadata() again live
       for the reference species (human, house mouse, chicken, zebrafish)

Requires network access to eutils.ncbi.nlm.nih.gov (unauthenticated —
no API key needed, though NCBI_API_KEY can be set as an env var to get a
higher eutils rate limit if you have one).

Run every check for the default species (tiger):
    python -m genome_agent.scripts.run_ncbi_live_check

Check a different species:
    python -m genome_agent.scripts.run_ncbi_live_check --species "house mouse"

Check a species you expect NOT to resolve, to see the failure path:
    python -m genome_agent.scripts.run_ncbi_live_check --species "dragon"

Skip the sequence window fetch (it downloads actual sequence bytes):
    python -m genome_agent.scripts.run_ncbi_live_check --skip-sequence

Skip the size_comparison visualization (it makes 3-4 extra live lookups):
    python -m genome_agent.scripts.run_ncbi_live_check --skip-comparison

or, from inside the container:
    python scripts/run_ncbi_live_check.py --species tiger
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Any

# Node-level INFO logs from the subagents are useful in isolation but
# drown out the trace below.
logging.getLogger().setLevel(logging.WARNING)

from ..subagents.gene_annotation import get_gene_annotation
from ..subagents.genome_metadata import get_genome_metadata
from ..subagents.sequence_window import (
    MAX_WINDOW_BP,
    NoLinkedSequenceError,
    WindowTooLargeError,
    fetch_sequence_window,
)
from ..subagents.species_resolver import resolve_species
from ..subagents.visualization import generate_visualization

_WIDTH = 64


def _header(title: str) -> None:
    print()
    print("=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def _kv(label: str, value: Any) -> None:
    print(f"  {label:<20}: {value}")


async def _check_species_resolver(species_name: str) -> dict | None:
    _header(f"1. species_resolver.resolve_species({species_name!r})")
    t0 = time.monotonic()
    try:
        result = await resolve_species(species_name)
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return None
    elapsed = time.monotonic() - t0

    _kv("assembly_id", result.get("assembly_id"))
    _kv("scientific_name", result.get("scientific_name"))
    _kv("common_name", result.get("common_name"))
    _kv("confidence", result.get("confidence"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if result.get("assembly_id") is None:
        print("  -> species did not resolve; downstream checks will be skipped.")
    return result


async def _check_genome_metadata(assembly_id: str) -> dict | None:
    _header(f"2. genome_metadata.get_genome_metadata({assembly_id!r})")
    t0 = time.monotonic()
    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return None
    elapsed = time.monotonic() - t0

    _kv("genome_size_bp", result.get("genome_size_bp"))
    _kv("chromosome_count", result.get("chromosome_count"))
    _kv("assembly_level", result.get("assembly_level"))
    _kv("karyotype", result.get("karyotype"))
    _kv("elapsed", f"{elapsed:.2f}s")
    return result


async def _check_gene_annotation(assembly_id: str) -> dict | None:
    _header(f"3. gene_annotation.get_gene_annotation({assembly_id!r})")
    t0 = time.monotonic()
    try:
        result = await get_gene_annotation(assembly_id)
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return None
    elapsed = time.monotonic() - t0

    gene_list = result.get("gene_list", [])
    _kv("gene count", len(gene_list))
    _kv("first 5 genes", gene_list[:5])
    if result.get("gene_table"):
        _kv("sample row", result["gene_table"][0])
    _kv("elapsed", f"{elapsed:.2f}s")
    return result


async def _check_sequence_window(assembly_id: str) -> None:
    _header(f"4. sequence_window.fetch_sequence_window({assembly_id!r}, 1, 500)")
    t0 = time.monotonic()
    try:
        seq = await fetch_sequence_window(assembly_id, 1, 500)
    except NoLinkedSequenceError as exc:
        print(f"  No linked Nuccore sequence found: {exc}")
        return
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return
    elapsed = time.monotonic() - t0

    _kv("bytes returned", len(seq))
    _kv("preview", seq[:80].replace("\n", " "))
    _kv("elapsed", f"{elapsed:.2f}s")

    # Also demonstrate the safety guard raises before any HTTP call.
    print("  Confirming MAX_WINDOW_BP guard (should raise, no HTTP call)...")
    try:
        await fetch_sequence_window(assembly_id, 0, MAX_WINDOW_BP + 1)
        print("  UNEXPECTED: did not raise")
    except WindowTooLargeError as exc:
        print(f"  OK — correctly raised: {exc}")


async def _check_size_comparison(species: dict, metadata: dict) -> None:
    _header("5. visualization.generate_visualization(scope='size_comparison')")
    t0 = time.monotonic()
    try:
        result = await generate_visualization(
            scope="size_comparison",
            genome_size_bp=metadata.get("genome_size_bp"),
            assembly_id=species.get("assembly_id"),
            common_name=species.get("common_name"),
            scientific_name=species.get("scientific_name"),
        )
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return
    elapsed = time.monotonic() - t0

    _kv("status", result.get("status"))
    _kv("format", result.get("format"))
    if result.get("chart_data"):
        _kv("chart_data bytes", len(result["chart_data"]))
    if result.get("note"):
        _kv("note", result["note"])
    for c in result.get("comparisons") or []:
        marker = " <- queried" if c.get("is_queried_species") else ""
        gb = c["genome_size_bp"] / 1_000_000_000
        print(f"    - {c['common_name']} ({c['scientific_name']}): {gb:.2f} Gb{marker}")
    _kv("elapsed", f"{elapsed:.2f}s")


async def run(species_name: str, skip_sequence: bool, skip_comparison: bool) -> None:
    species = await _check_species_resolver(species_name)
    if not species or not species.get("assembly_id"):
        print("\nNo assembly_id resolved — nothing further to check.")
        return

    assembly_id = species["assembly_id"]

    metadata = await _check_genome_metadata(assembly_id)
    await _check_gene_annotation(assembly_id)

    if not skip_sequence:
        await _check_sequence_window(assembly_id)
    else:
        print("\n(skipping sequence_window check — --skip-sequence)")

    if not skip_comparison:
        if metadata:
            await _check_size_comparison(species, metadata)
        else:
            print("\n(skipping size_comparison check — no metadata to compare)")
    else:
        print("\n(skipping size_comparison check — --skip-comparison)")

    _header("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="tiger",
        help="Species name to resolve and look up (default: tiger)",
    )
    parser.add_argument(
        "--skip-sequence",
        action="store_true",
        help="Skip the sequence_window fetch (downloads real sequence bytes)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the size_comparison visualization (extra live lookups)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.species, args.skip_sequence, args.skip_comparison))


if __name__ == "__main__":
    main()
