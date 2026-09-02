"""
Canned fixtures for the Part B orchestration eval (agent_test_cases.yaml /
agent_capture.py / agent_evaluators.py).

Every fixture here is a *fake* replacement for exactly the module-level names
the existing test suite already monkeypatches (see
tests/test_functional_evidence_partial_failure_propagation.py and
scripts/literature_support_scenarios.py):

    workflows/nodes/gene_mapper_node.py           -> gene_mapper_agent
    workflows/nodes/functional_evidence_nodes.py  -> pathways_agent, protein_data_agent
    workflows/nodes/literature_support_node.py    -> literature_support_agent
    workflows/nodes/escalation_nodes.py            -> resolve_capability (imported
                                                       as workflows.capability_resolver.resolve_capability)
    workflows/trait_discovery_graph.py             -> write_explanation

Patching at this boundary (not inside subagents/*) means the *graph itself*
— every node, edge, and conditional route in trait_discovery_graph.py and
functional_evidence_graph.py — runs completely unmodified. That's the whole
point of a Part B eval: we're scoring orchestration behavior, not retrieval
quality (that's Part A), so the only thing that may ever differ from a real
run is what a subagent *decided*, never how the graph reacts to it.

Each fixture optionally takes `sleep_seconds` so check_parallelism (Step 5)
can tell concurrent execution apart from sequential: two branches that each
sleep ~0.15s and together take ~0.15s (not ~0.3s) really ran in parallel.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from schemas.common import AgentStatus
from schemas.outputs import (
    GeneMapperOutput,
    GOAnnotation,
    LiteratureRecord,
    LiteratureSupportOutput,
    PathwayEntry,
    PathwaysOutput,
    ProteinDataOutput,
    ProteinEntry,
)
from workflows.capability_resolver import CapabilityResolution

DEFAULT_SLEEP = 0.15  # long enough to be measurable, short enough to keep the eval fast


@dataclass
class TimingLog:
    """Shared sink every timed fixture writes (label, start, end) into, so
    check_parallelism can compare wall-clock overlap across branches without
    touching LangGraph's own scheduling internals."""
    spans: list[tuple[str, float, float]] = field(default_factory=list)

    def record(self, label: str, start: float, end: float) -> None:
        self.spans.append((label, start, end))

    def synthesize(self, label: str, from_labels: list[str]) -> None:
        """Derive a span for a composite stage (e.g. 'functional_evidence',
        which is a subgraph, not a single fixture call) by enclosing the
        min-start/max-end of its constituent spans. No-op if any constituent
        never ran."""
        spans = [s for s in self.spans if s[0] in from_labels]
        if len(spans) != len(from_labels):
            return
        start = min(s[1] for s in spans)
        end = max(s[2] for s in spans)
        self.record(label, start, end)

    def overlap_seconds(self, label_a: str, label_b: str) -> float | None:
        """Positive overlap = ran concurrently. 0 or negative = sequential
        (or one of the labels never ran, e.g. an escalation short-circuited
        it) -> returns None in that case."""
        a = next((s for s in self.spans if s[0] == label_a), None)
        b = next((s for s in self.spans if s[0] == label_b), None)
        if a is None or b is None:
            return None
        _, a_start, a_end = a
        _, b_start, b_end = b
        return min(a_end, b_end) - max(a_start, b_start)


async def _timed(label: str, timing_log: TimingLog | None, sleep_seconds: float, value):
    start = time.perf_counter()
    if sleep_seconds:
        await asyncio.sleep(sleep_seconds)
    end = time.perf_counter()
    if timing_log is not None:
        timing_log.record(label, start, end)
    return value


# ---------------------------------------------------------------------------
# gene_mapper_agent fakes
# ---------------------------------------------------------------------------
def gene_mapper_completed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        annotations = [GOAnnotation(gene_symbol=g, go_id=f"GO:{1000 + i}", go_name=f"mock function of {g}")
                       for i, g in enumerate(input.gene_list)]
        return await _timed("gene_mapper", timing_log, sleep_seconds,
                             GeneMapperOutput(status=AgentStatus.COMPLETED, go_annotations=annotations))
    return _fake


def gene_mapper_failed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        return await _timed("gene_mapper", timing_log, sleep_seconds,
                             GeneMapperOutput(status=AgentStatus.FAILED, unmatched_genes=list(input.gene_list)))
    return _fake


# ---------------------------------------------------------------------------
# pathways_agent fakes
# ---------------------------------------------------------------------------
def pathways_completed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        entries = [PathwayEntry(pathway_id=f"hsa{4000 + i}", pathway_name=f"mock pathway for {g}")
                   for i, g in enumerate(input.gene_list)]
        return await _timed("pathways", timing_log, sleep_seconds,
                             PathwaysOutput(status=AgentStatus.COMPLETED, pathways=entries))
    return _fake


def pathways_failed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        return await _timed("pathways", timing_log, sleep_seconds,
                             PathwaysOutput(status=AgentStatus.FAILED, malformed_ids=list(input.gene_list)))
    return _fake


# ---------------------------------------------------------------------------
# protein_data_agent fakes
# ---------------------------------------------------------------------------
def protein_data_completed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        entries = [ProteinEntry(gene_symbol=g, protein_name=f"mock protein for {g}",
                                 function_summary=f"mock function summary for {g}", source_accession=f"P{i:05d}")
                   for i, g in enumerate(input.gene_list)]
        return await _timed("protein_data", timing_log, sleep_seconds,
                             ProteinDataOutput(status=AgentStatus.COMPLETED, proteins=entries))
    return _fake


def protein_data_failed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        return await _timed("protein_data", timing_log, sleep_seconds,
                             ProteinDataOutput(status=AgentStatus.FAILED, missing_genes=list(input.gene_list)))
    return _fake


# ---------------------------------------------------------------------------
# literature_support_agent fakes
# ---------------------------------------------------------------------------
def literature_completed(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        evidence = [LiteratureRecord(pmid="18239092", title="mock literature record", year=2008,
                                      short_summary=f"mock evidence for {', '.join(input.gene_list)}")]
        return await _timed("literature_support", timing_log, sleep_seconds,
                             LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=evidence))
    return _fake


def literature_thin_needs_agent(timing_log: TimingLog | None = None, sleep_seconds: float = DEFAULT_SLEEP):
    async def _fake(input):
        return await _timed(
            "literature_support", timing_log, sleep_seconds,
            LiteratureSupportOutput(
                status=AgentStatus.NEEDS_AGENT,
                evidence=[],
                target_agent="Literature Agent",
                prompt_to_target_agent=f"Find deeper evidence for {input.trait_name}.",
            ),
        )
    return _fake


# ---------------------------------------------------------------------------
# resolve_capability fakes (workflows.capability_resolver.resolve_capability,
# patched at workflows.nodes.escalation_nodes.resolve_capability)
# ---------------------------------------------------------------------------
def resolver_returns(target_agent: str, prompt: str = "mock prompt to target agent", reasoning: str = "mock reasoning"):
    async def _fake(*, waiting_agent, need_description, known_context):
        return CapabilityResolution(target_agent=target_agent, prompt_to_target_agent=prompt, reasoning=reasoning)
    return _fake


def resolver_must_not_be_called():
    async def _fake(*, waiting_agent, need_description, known_context):
        raise AssertionError(f"resolve_capability must not be called (waiting_agent={waiting_agent!r})")
    return _fake


# ---------------------------------------------------------------------------
# write_explanation fake (workflows.trait_discovery_graph.write_explanation)
# ---------------------------------------------------------------------------
def fake_write_explanation(call_log: list | None = None):
    async def _fake(**kwargs):
        if call_log is not None:
            call_log.append(dict(kwargs))
        return f"mock explanation for {kwargs['trait_name']}: genes {kwargs['genes']}"
    return _fake


def write_explanation_must_not_be_called():
    async def _fake(**kwargs):
        raise AssertionError("write_explanation must not be called on a failed/escalated run")
    return _fake
