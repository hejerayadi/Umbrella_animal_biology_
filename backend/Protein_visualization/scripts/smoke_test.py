import argparse
from uuid import uuid4

import httpx

from app.configuration.settings import get_settings


def main(base_url: str, accession: str, gene: str, taxon_id: int) -> None:
    task_id = str(uuid4())
    trace_id = str(uuid4())
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/v1/protein-structure-analyses",
        headers={"X-Trace-Id": trace_id},
        json={
            "task_id": task_id,
            "trace_id": trace_id,
            "idempotency_key": f"smoke-{task_id}",
            "input": {
                "resolved_gene_id": gene,
                "species": {"scientific_name": "Homo sapiens", "taxon_id": taxon_id},
                "uniprot_accession": accession,
            },
        },
        timeout=60,
    )
    body = response.json()
    if body.get("error"):
        error = body["error"]
        raise SystemExit(f"{error['code']}: {error['detail']} (trace_id={body['meta']['trace_id']})")

    agent_result = body["data"]
    data = agent_result["output"]
    print(
        f"analysis={data['analysis_id']} agent_status={agent_result['status']} "
        f"analysis_status={data['status']} "
        f"validation={data['validation_status']} duration_ms={body['meta']['duration_ms']}"
    )
    if agent_result["target_agent"]:
        print(f"target_agent={agent_result['target_agent']} prompt={agent_result['prompt_to_target_agent']}")


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=settings.local_base_url)
    parser.add_argument("--accession", default="P04637")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--taxon-id", type=int, default=9606)
    args = parser.parse_args()
    main(args.base_url, args.accession, args.gene, args.taxon_id)
