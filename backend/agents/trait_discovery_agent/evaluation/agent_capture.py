"""
Step 4 — runner for the Part B orchestration eval.

`run_case` builds and invokes the *real, unmodified* compiled
`build_trait_discovery_graph()` (same as tests/test_orchestrator_scenarios.py
and scripts/literature_support_scenarios.py) and patches only the module-level
subagent/resolver/writer names each case needs — never the graph, its nodes,
or its edges. See agent_fixtures.py's module docstring for exactly which
names get patched and why patching there (not inside subagents/*) leaves
orchestration behavior fully real.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workflows.nodes.escalation_nodes as esc_module  # noqa: E402
import workflows.nodes.functional_evidence_nodes as fe_nodes  # noqa: E402
import workflows.nodes.gene_mapper_node as gm_node  # noqa: E402
import workflows.nodes.literature_support_node as lit_node  # noqa: E402
import workflows.trait_discovery_graph as td_graph_module  # noqa: E402
from workflows.state import TraitDiscoveryState  # noqa: E402

import agent_fixtures  # noqa: E402
from agent_fixtures import TimingLog  # noqa: E402

AGENT_TEST_CASES_PATH = Path(__file__).parent / "agent_test_cases.yaml"

# Where each fixture key in a case's `fixtures:` block gets patched.
PATCH_TARGETS = {
    "gene_mapper": (gm_node, "gene_mapper_agent"),
    "pathways": (fe_nodes, "pathways_agent"),
    "protein_data": (fe_nodes, "protein_data_agent"),
    "literature_support": (lit_node, "literature_support_agent"),
    "resolve_capability": (esc_module, "resolve_capability"),
    "write_explanation": (td_graph_module, "write_explanation"),
}

# Fixtures that need a TimingLog + sleep_seconds threaded in; the rest
# (resolver/explanation fakes) take no such args.
TIMED_FIXTURE_NODES = {"gene_mapper", "pathways", "protein_data", "literature_support"}

# functional_evidence is a subgraph (pathways + protein_data), not a single
# fixture call, so its span is synthesized from its two children rather than
# recorded directly. See TimingLog.synthesize.
SYNTHESIZED_SPANS = {"functional_evidence": ["pathways", "protein_data"]}


def load_cases(path: Path = AGENT_TEST_CASES_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_fixture_spec(node_key: str, spec: Any, timing_log: TimingLog) -> Any:
    """A case's fixture entry is either:
      - null                          -> don't patch this name at all
      - "builder_name"                -> agent_fixtures.builder_name(...)
      - {fixture: "builder_name", ...extra kwargs merged in (e.g. sleep_seconds,
        target_agent)}
    Returns the fake coroutine to patch in, or None if nothing should be patched.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        builder_name, kwargs = spec, {}
    elif isinstance(spec, dict):
        spec = dict(spec)
        builder_name = spec.pop("fixture")
        kwargs = spec
    else:
        raise ValueError(f"Unrecognized fixture spec for {node_key!r}: {spec!r}")

    builder = getattr(agent_fixtures, builder_name)
    if node_key in TIMED_FIXTURE_NODES:
        kwargs.setdefault("timing_log", timing_log)
    return builder(**kwargs)


@dataclass
class CaseRun:
    case_id: str
    node_order: list[str] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    timing_log: TimingLog = field(default_factory=TimingLog)
    aggregate_calls: list = field(default_factory=list)  # kwargs write_explanation was actually called with
    error: str | None = None


async def run_case(case: dict) -> CaseRun:
    run = CaseRun(case_id=case["id"])
    timing_log = run.timing_log

    fixtures = case.get("fixtures", {})
    patches = []
    for node_key, spec in fixtures.items():
        if node_key not in PATCH_TARGETS:
            raise ValueError(f"Case {case['id']!r} references unknown fixture target {node_key!r}")
        if node_key == "write_explanation" and spec == "fake_write_explanation":
            fake = agent_fixtures.fake_write_explanation(call_log=run.aggregate_calls)
        else:
            fake = _resolve_fixture_spec(node_key, spec, timing_log)
        if fake is None:
            continue
        target_module, attr_name = PATCH_TARGETS[node_key]
        patches.append(patch.object(target_module, attr_name, fake))

    input_spec = case["input"]
    state = TraitDiscoveryState(
        trait_name=input_spec["trait_name"],
        species_name=input_spec["species_name"],
        instruction=input_spec["instruction"],
        context=input_spec.get("context", {}),
    )

    try:
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            # Rebuild the graph *inside* the patch context: build_trait_discovery_graph()
            # binds gene_mapper_agent/pathways_agent/etc. as default args / closures at
            # node-registration time in some LangGraph versions, so patching after the
            # graph is already built can silently miss. Building fresh per case also
            # matches how every existing test in this repo does it (see
            # test_orchestrator_scenarios.py) rather than reusing a module-level graph.
            app = td_graph_module.build_trait_discovery_graph()

            accumulated: dict[str, Any] = {}
            async for update in app.astream(state, stream_mode="updates"):
                (node_name, payload), = update.items()
                run.node_order.append(node_name)
                accumulated.update(payload)
            run.final_state = accumulated
            for label, from_labels in SYNTHESIZED_SPANS.items():
                timing_log.synthesize(label, from_labels)
    except Exception as exc:  # noqa: BLE001 - eval harness: capture, don't crash the whole run
        run.error = f"{type(exc).__name__}: {exc}"

    return run
