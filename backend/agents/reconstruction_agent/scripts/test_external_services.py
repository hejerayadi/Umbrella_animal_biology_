"""Check that NCBI and EMBL-EBI are reachable and correctly configured.

This one DOES hit the real services. Run it when setting the agent up, or when
diagnosing a deployment that starts cleanly but fails every reconstruction.

    uv run python scripts/test_external_services.py

BLAST and MAFFT submit real jobs and can take a minute or two each.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconstruction_agent.configuration.logging import configure_logging  # noqa: E402
from reconstruction_agent.configuration.settings import get_settings  # noqa: E402
from reconstruction_agent.infrastructure.embl_ebi.blast_client import (  # noqa: E402
    BlastClient,
)
from reconstruction_agent.infrastructure.embl_ebi.mafft_client import (  # noqa: E402
    MafftClient,
)
from reconstruction_agent.infrastructure.ncbi.client import NCBIClient  # noqa: E402
from reconstruction_agent.tools.ncbi.mapper import to_references  # noqa: E402

# Short enough that BLAST and MAFFT return quickly.
QUERY = "ACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCA"


async def check_ncbi() -> bool:
    settings = get_settings()
    print("\n=== NCBI E-utilities ===")
    print(f"Rate budget: {settings.ncbi.requests_per_second}/s "
          f"({'keyed' if settings.ncbi.api_key else 'anonymous'})")

    client = NCBIClient(settings.ncbi)
    try:
        identifiers = await client.search("Loxodonta africana[Organism]", limit=2)
        if not identifiers:
            print("FAIL: search returned no identifiers.")
            return False
        print(f"  search  -> {identifiers}")

        fasta = await client.fetch_fasta(identifiers[:1])
        references = to_references(fasta)
        if not references or not references[0].residues:
            print("FAIL: efetch returned no residues.")
            return False
        print(f"  efetch  -> {references[0].accession} "
              f"({len(references[0].residues or '')} bases, {references[0].organism})")
        return True

    except Exception as error:  # noqa: BLE001 - this script reports, never raises
        print(f"FAIL: {error}")
        return False
    finally:
        await client.aclose()


async def check_blast() -> bool:
    settings = get_settings()
    print("\n=== EMBL-EBI BLAST ===")
    if not settings.embl_ebi.contact_email:
        print("SKIP: EMBL_EBI_CONTACT_EMAIL is not set; EMBL-EBI rejects anonymous jobs.")
        return False

    client = BlastClient(settings.embl_ebi)
    try:
        job_id = await client.submit(QUERY, max_hits=5)
        print(f"  submitted -> {job_id} (polling, this can take a minute)")
        raw = await client.result(job_id)
        print(f"  result    -> {len(raw)} bytes")
        return True
    except Exception as error:  # noqa: BLE001
        print(f"FAIL: {error}")
        return False
    finally:
        await client.aclose()


async def check_mafft() -> bool:
    settings = get_settings()
    print("\n=== EMBL-EBI MAFFT ===")
    if not settings.embl_ebi.contact_email:
        print("SKIP: EMBL_EBI_CONTACT_EMAIL is not set.")
        return False

    client = MafftClient(settings.embl_ebi)
    fasta = f">seq_a\n{QUERY}\n>seq_b\n{QUERY.replace('ACGT', 'ACGA', 1)}\n"
    try:
        job_id = await client.submit(fasta)
        print(f"  submitted -> {job_id} (polling)")
        aligned = await client.result(job_id)
        print(f"  result    -> {len(aligned)} bytes")
        return True
    except Exception as error:  # noqa: BLE001
        print(f"FAIL: {error}")
        return False
    finally:
        await client.aclose()


async def main() -> int:
    configure_logging()
    settings = get_settings()

    print("Reconstruction Agent - external service check")
    print(f"LLM: {'enabled (' + settings.llm.model + ')' if settings.llm.enabled else 'disabled'}")

    results = {
        "NCBI": await check_ncbi(),
        "BLAST": await check_blast(),
        "MAFFT": await check_mafft(),
    }

    print("\n=== Summary ===")
    for service, ok in results.items():
        print(f"  {service:<6} {'OK' if ok else 'NOT AVAILABLE'}")

    if not results["NCBI"]:
        print("\nNCBI is required. The agent cannot fetch references without it.")
        return 1
    if not (results["BLAST"] and results["MAFFT"]):
        print(
            "\nBLAST and/or MAFFT are unavailable: the agent will detect gaps but "
            "report them unresolved. Set EMBL_EBI_CONTACT_EMAIL in .env."
        )
        return 1

    print("\nAll external services reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
