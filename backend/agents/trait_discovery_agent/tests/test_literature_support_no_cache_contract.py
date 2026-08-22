"""
Contract test for design guide §6: Literature Support must NEVER cache
evidence content (title/short_summary) — only the scalar trait->gene->pmid
relationship goes to Neo4j.

This used to be true only because nothing was implemented (see the
"KEGG license"-style finding in this repo's own review notes: "not enforced
because it isn't implemented at all"). Now that write_trait_gene_relationship
is a real Neo4j write, the risk flips: it would be easy for a future edit to
also start caching evidence into Qdrant (e.g. "while we're at it, cache the
literature summaries too") without anyone noticing it violates the rule.

This test makes that violation fail CI immediately, in two ways:
  1. Static source scan — subagents/literature_support/*.py and
     kb/sources/literature_agent_client.py must never reference Qdrant's
     cache functions or import kb.qdrant_store at all. A rule enforced only
     by "we didn't happen to call it" isn't enforced; a rule that fails the
     build the moment someone imports the forbidden module is.
  2. Behavioral check — running literature_support_agent end-to-end (LLM
     mocked) with kb.qdrant_store.upsert_point/get_cached patched to raise
     confirms no code path anywhere underneath it (including the bind_tools
     loop) reaches those functions.
"""
import ast
from pathlib import Path

import pytest

from schemas.inputs import LiteratureSupportInput
from schemas.common import AgentStatus
import kb.qdrant_store as qdrant_store
import subagents.literature_support as ls_module

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCANNED_FILES = [
    _REPO_ROOT / "subagents" / "literature_support" / "__init__.py",
    _REPO_ROOT / "subagents" / "literature_support" / "llm_pick.py",
    _REPO_ROOT / "subagents" / "literature_support" / "mock.py",
    _REPO_ROOT / "kb" / "sources" / "literature_agent_client.py",
]
_FORBIDDEN_NAMES = {"qdrant_store", "upsert_point", "get_cached", "embed_text"}


def _names_referenced(source: str) -> set[str]:
    """All bare names and attribute-access roots referenced anywhere in the
    module — catches `import kb.qdrant_store`, `from kb.qdrant_store import
    upsert_point`, and `qdrant_store.upsert_point(...)` alike."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
    return names


def test_no_qdrant_references_in_literature_support_source():
    for path in _SCANNED_FILES:
        assert path.exists(), f"expected file not found: {path}"
        referenced = _names_referenced(path.read_text())
        hit = referenced & _FORBIDDEN_NAMES
        assert not hit, (
            f"{path.relative_to(_REPO_ROOT)} references {hit} — Literature "
            "Support must never touch the Qdrant cache (design guide §6). "
            "If this is intentional, the rule itself needs updating, not "
            "just this test."
        )


@pytest.mark.asyncio
async def test_agent_run_never_touches_qdrant_even_when_llm_writes_evidence(monkeypatch):
    async def _fake_fetch(trait, genes):
        return [
            {"pmid": "18239092", "title": "FGF5 and hair cycle regulation", "year": 2008,
             "short_summary": "Links FGF5 mutation to hair length in mammals."},
            {"pmid": "30112233", "title": "Follicle regulatory network in rodents", "year": 2019,
             "short_summary": "Broader gene network context around FGF5 signaling."},
        ]

    async def _fake_judge(trait, genes, evidence, thin_flag):
        return True, ["18239092"], "clear FGF5 link", ""

    async def _forbidden(*args, **kwargs):
        raise AssertionError(
            "Literature Support must never call the Qdrant cache — evidence "
            "content is not allowed to be cached (design guide §6)."
        )

    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch)
    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", _fake_judge)
    monkeypatch.setattr(qdrant_store, "upsert_point", _forbidden)
    monkeypatch.setattr(qdrant_store, "get_cached", _forbidden)

    result = await ls_module.literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2
