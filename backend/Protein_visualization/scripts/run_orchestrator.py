"""Run the real Protein Orchestrator in-process for local diagnostics.

Every biological client uses its configured real provider. This command only
constructs the ``AgentTask`` that the future Umbrella Grand Orchestrator will send.
Use ``scripts/smoke_test.py`` to exercise the HTTP path instead.
"""

import argparse
import asyncio
import json
from uuid import uuid4

from app.api.v1.dependencies import get_orchestrator
from app.configuration.settings import get_settings
from app.contracts.agent_task import AgentTask
from app.observability.logging import configure_logging


def build_task(accession: str | None, gene: str, taxon_id: int, residue: int | None) -> AgentTask:
    task_id = uuid4()
    return AgentTask.model_validate(
        {
            "task_id": str(task_id),
            "trace_id": str(uuid4()),
            "source_agent": "local-orchestrator-runner",
            "target_capability": "protein-structure.orchestrate",
            "schema_version": "1.0",
            "idempotency_key": f"local-{task_id}",
            "input": {
                "resolved_gene_id": gene,
                "species": {"scientific_name": "Homo sapiens", "taxon_id": taxon_id},
                "uniprot_accession": accession,
                "residue_position": residue,
                "preferred_source": "AUTO",
                "include_explanation": True,
            },
        }
    )


async def main(task: AgentTask) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    result = await get_orchestrator().execute(task)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", default="P04637")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--taxon-id", type=int, default=9606)
    parser.add_argument("--residue", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(build_task(args.accession, args.gene, args.taxon_id, args.residue)))
