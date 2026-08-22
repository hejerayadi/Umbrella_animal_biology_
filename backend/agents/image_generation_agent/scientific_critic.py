"""Scientific critic for protein illustration evidence."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .tool_schemas import (
    CriticResult,
    CriticVerdict,
    EvidenceBundle,
    PDBStructureResult,
    UniProtResult,
    WebSearchResult,
)

logger = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR / ".env", override=True)

SEVERITY = {"ACCEPT": 0, "REVISE": 1, "ABSTAIN": 2}

SCIENTIFIC_CRITIC_SYSTEM_PROMPT = """You are a scientific critic auditing evidence for a protein illustration request.

You receive JSON with UniProt, PDB, web search, and user context. Evaluate protein identity,
database evidence, and any web snippets. Never invent accession numbers, structures, or facts.

Return one verdict:
- ACCEPT: identity and evidence are coherent and sufficient for a grounded illustration.
- REVISE: illustration may proceed but evidence is incomplete or needs caveats.
- ABSTAIN: protein identity is ambiguous or evidence is insufficient to illustrate faithfully.

Give short factual reasons grounded only in the supplied evidence."""


class ScientificCritic:
    """Deterministic critic with optional Azure gpt-4.1-mini audit via v1/responses."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else os.getenv("AZURE_OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_CRITIC_DEPLOYMENT", "gpt-4.1-mini")
        self.api_version = api_version or os.getenv("AZURE_CRITIC_API_VERSION", "2024-10-21")
        self.timeout = timeout or int(os.getenv("AZURE_CRITIC_TIMEOUT", "60"))
        self._session = session or requests.Session()

    @property
    def llm_enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.deployment)

    def review(
        self,
        *,
        gene: str,
        species: str | None,
        instruction: str,
        uniprot: UniProtResult | None,
        pdb: PDBStructureResult | None,
        web_search: WebSearchResult | None,
    ) -> CriticResult:
        deterministic = _deterministic_verdict(
            gene=gene,
            species=species,
            uniprot=uniprot,
            pdb=pdb,
            web_search=web_search,
        )
        if not self.llm_enabled:
            return deterministic

        try:
            llm_result = self._llm_audit(
                deterministic=deterministic,
                gene=gene,
                species=species,
                instruction=instruction,
                uniprot=uniprot,
                pdb=pdb,
                web_search=web_search,
            )
        except Exception as exc:
            logger.warning("Scientific critic LLM audit failed: %s", type(exc).__name__)
            return deterministic

        return _strictest(deterministic, llm_result)

    def _llm_audit(
        self,
        *,
        deterministic: CriticResult,
        gene: str,
        species: str | None,
        instruction: str,
        uniprot: UniProtResult | None,
        pdb: PDBStructureResult | None,
        web_search: WebSearchResult | None,
    ) -> CriticResult:
        context = _build_audit_context(
            deterministic=deterministic,
            gene=gene,
            species=species,
            instruction=instruction,
            uniprot=uniprot,
            pdb=pdb,
            web_search=web_search,
        )
        url = f"{self.base_url}/openai/v1/responses"
        payload = {
            "model": self.deployment,
            "input": [
                {"role": "system", "content": SCIENTIFIC_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, default=str)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "scientific_critic_verdict",
                    "strict": True,
                    "schema": _critic_json_schema(),
                }
            },
            "store": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        params = {"api-version": self.api_version}
        response = self._session.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _parse_llm_verdict(_extract_response_text(response.json()))


def scientific_critic(
    bundle: EvidenceBundle,
    *,
    gene: str,
    species: str | None,
    instruction: str,
    critic: ScientificCritic | None = None,
) -> CriticResult:
    """Evaluate gathered evidence and return ACCEPT / REVISE / ABSTAIN."""
    reviewer = critic or ScientificCritic()
    return reviewer.review(
        gene=gene,
        species=species,
        instruction=instruction,
        uniprot=bundle.uniprot,
        pdb=bundle.pdb,
        web_search=bundle.web_search,
    )


def _deterministic_verdict(
    *,
    gene: str,
    species: str | None,
    uniprot: UniProtResult | None,
    pdb: PDBStructureResult | None,
    web_search: WebSearchResult | None,
) -> CriticResult:
    verdict: CriticVerdict = "ACCEPT"
    reasons: list[str] = []

    if uniprot is None or not uniprot.success:
        return CriticResult(
            verdict="ABSTAIN",
            reasons=(
                uniprot.error if uniprot and uniprot.error else f"Protein identity for '{gene}' was not confirmed.",
            ),
            source="deterministic",
        )

    reasons.append(
        f"UniProt accession {uniprot.accession} resolved for '{gene}'"
        + (f" ({uniprot.protein_name})" if uniprot.protein_name else "")
        + "."
    )

    if species and uniprot.organism and _organisms_conflict(species, uniprot.organism):
        verdict = "REVISE"
        reasons.append(
            f"Requested species '{species}' differs from UniProt organism '{uniprot.organism}'."
        )

    if pdb is None or not pdb.found:
        verdict = _strictest_verdict(verdict, "REVISE")
        message = pdb.message if pdb and pdb.message else "No experimental PDB structure was selected."
        reasons.append(message)
    else:
        reasons.append(
            f"PDB structure {pdb.pdb_id} selected ({pdb.experimental_method or 'method unknown'})."
        )
        if pdb.sequence_coverage is not None and pdb.sequence_coverage < 1.0:
            verdict = _strictest_verdict(verdict, "REVISE")
            reasons.append(
                f"Structure covers {pdb.sequence_coverage:.0%} of the UniProt sequence."
            )

    if web_search and web_search.success and web_search.results:
        reasons.append(f"Web search returned {len(web_search.results)} supplementary result(s).")
    elif web_search and web_search.error:
        reasons.append(f"Web search unavailable: {web_search.error}")

    if verdict == "ACCEPT" and len(reasons) == 1:
        reasons.append("Identity and structural evidence are mutually consistent.")

    return CriticResult(verdict=verdict, reasons=tuple(reasons), source="deterministic")


def _organisms_conflict(requested: str, resolved: str) -> bool:
    requested_norm = requested.strip().lower()
    resolved_norm = resolved.strip().lower()
    if not requested_norm or not resolved_norm:
        return False
    return requested_norm not in resolved_norm and resolved_norm not in requested_norm


def _strictest_verdict(current: CriticVerdict, proposed: CriticVerdict) -> CriticVerdict:
    return proposed if SEVERITY[proposed] > SEVERITY[current] else current


def _strictest(deterministic: CriticResult, llm_result: CriticResult) -> CriticResult:
    current = deterministic.verdict
    proposed = llm_result.verdict
    if SEVERITY[proposed] <= SEVERITY[current]:
        return deterministic
    merged_reasons = tuple(dict.fromkeys((*deterministic.reasons, *llm_result.reasons)))
    return CriticResult(verdict=proposed, reasons=merged_reasons, source="llm")


def _build_audit_context(
    *,
    deterministic: CriticResult,
    gene: str,
    species: str | None,
    instruction: str,
    uniprot: UniProtResult | None,
    pdb: PDBStructureResult | None,
    web_search: WebSearchResult | None,
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "gene": gene,
        "species": species,
        "deterministic_verdict": deterministic.verdict,
        "deterministic_reasons": list(deterministic.reasons),
        "uniprot": _serialize_uniprot(uniprot),
        "pdb": _serialize_pdb(pdb),
        "web_search": _serialize_web_search(web_search),
    }


def _serialize_uniprot(result: UniProtResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "success": result.success,
        "accession": result.accession,
        "protein_name": result.protein_name,
        "organism": result.organism,
        "sequence_length": result.sequence_length,
        "error": result.error,
    }


def _serialize_pdb(result: PDBStructureResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "found": result.found,
        "pdb_id": result.pdb_id,
        "experimental_method": result.experimental_method,
        "resolution": result.resolution,
        "sequence_coverage": result.sequence_coverage,
        "title": result.title,
        "organism": result.organism,
        "chain": result.chain,
        "selection_reason": result.selection_reason,
        "message": result.message,
    }


def _serialize_web_search(result: WebSearchResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "success": result.success,
        "error": result.error,
        "results": [
            {"title": hit.title, "link": hit.link, "snippet": hit.snippet}
            for hit in result.results
        ],
    }


def _critic_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["ACCEPT", "REVISE", "ABSTAIN"]},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "reasons"],
        "additionalProperties": False,
    }


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    raise ValueError("Azure responses payload did not contain output text.")


def _parse_llm_verdict(text: str) -> CriticResult:
    data = json.loads(text)
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in SEVERITY:
        raise ValueError(f"Unexpected critic verdict: {verdict!r}")
    reasons_raw = data.get("reasons") or []
    reasons = tuple(str(reason).strip() for reason in reasons_raw if str(reason).strip())
    return CriticResult(verdict=verdict, reasons=reasons, source="llm")

