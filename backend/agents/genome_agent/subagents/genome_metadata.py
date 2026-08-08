"""
Genome Metadata — real NCBI Assembly eutils subagent (Task 5)
Retrieves genome size, chromosome count, and assembly level for a resolved assembly.
Never cached — pass-through only, cheap to refetch.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET

from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)


def _parse_meta_stats(meta_xml: str) -> dict[str, str]:
    """Parse the <Stats> block inside NCBI Assembly esummary 'meta' field."""
    stats: dict[str, str] = {}
    if not meta_xml:
        return stats
    try:
        import re
        m = re.search(r"<Stats>(.*?)</Stats>", meta_xml, re.DOTALL)
        if not m:
            return stats
        stats_xml = m.group(1)
        wrapped = f"<root>{stats_xml}</root>"
        root = ET.fromstring(wrapped)
        for stat in root.findall("Stat"):
            category = stat.get("category")
            if category:
                stats[category] = (stat.text or "").strip()
    except ET.ParseError as exc:
        logger.warning("Failed to parse assembly meta XML: %s", exc)
    return stats


async def _resolve_assembly_uid(assembly_id: str) -> str | None:
    """Convert an assembly accession (e.g. GCA_024034525.1) to a numeric NCBI UID."""
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": f"{assembly_id}[Assembly]",
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])
    return uid_list[0] if uid_list else None


async def get_genome_metadata(assembly_id: str) -> dict:
    uid = await _resolve_assembly_uid(assembly_id)
    if not uid:
        return {
            "genome_size_bp": None,
            "chromosome_count": None,
            "karyotype": None,
            "assembly_level": None,
        }

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})

    meta_xml = assembly_info.get("meta", "")
    stats = _parse_meta_stats(meta_xml)

    genome_size_bp = stats.get("total_length")
    if genome_size_bp is not None:
        try:
            genome_size_bp = int(genome_size_bp)
        except ValueError:
            genome_size_bp = None

    chromosome_count = stats.get("chromosome_count")
    if chromosome_count is not None:
        try:
            chromosome_count = int(chromosome_count)
        except ValueError:
            chromosome_count = None

    assembly_level = assembly_info.get("assemblystatus") or stats.get("assembly-level")

    return {
        "genome_size_bp": genome_size_bp,
        "chromosome_count": chromosome_count,
        "karyotype": None,
        "assembly_level": assembly_level,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Genome Metadata live NCBI test ---")
        result = await get_genome_metadata("GCA_024034525.1")
        print("Tiger assembly metadata:", result)
        assert result.get("genome_size_bp") is not None, "Expected genome_size_bp from NCBI"
        assert result.get("chromosome_count") is not None, "Expected chromosome_count from NCBI"
        assert result.get("assembly_level") is not None, "Expected assembly_level from NCBI"
        print("All tests passed ✅")

    asyncio.run(_quick_test())
