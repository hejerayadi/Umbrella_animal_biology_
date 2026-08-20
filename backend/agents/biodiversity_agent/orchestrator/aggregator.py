"""Aggregation logic for the Biodiversity Orchestrator.

When the orchestrator dispatches multiple workers in parallel (e.g.
distribution AND migration for the same species), we get a list of
``AgentResult``s back. This module packages them into a single
``AgentResult`` that the Global Orchestrator can consume unchanged.

Rules:

- If any child returned ``NEEDS_AGENT``, the aggregate result is also
  ``NEEDS_AGENT`` - we do not swallow escalations, we bubble the first
  one upward.
- If all children FAILED, the aggregate is FAILED.
- Otherwise the aggregate is COMPLETED with:
    * ``output``  - dict keyed by feature name.
    * ``map_url`` - the first non-null map_url (or a synthesized combined URL).
    * ``hotspots`` / ``migration_route`` / ``observation_count`` filled
      from the child that produced them (at most one is expected per
      feature so no merging is needed).
    * ``source_agents`` - flattened list of every source that participated.
"""

from __future__ import annotations

from typing import Iterable

from ..schema import AgentResult, AgentStatus, BiodiversityFeature


def aggregate(
    results: Iterable[tuple[BiodiversityFeature, AgentResult]],
) -> AgentResult:
    results_list = list(results)
    if not results_list:
        return AgentResult(
            status=AgentStatus.FAILED,
            output="No worker produced a result.",
            source_agents=["Biodiversity Agent Orchestrator"],
        )

    # 1. Bubble up escalations.
    for feature, r in results_list:
        if r.status == AgentStatus.NEEDS_AGENT:
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent=r.target_agent,
                prompt_to_target_agent=r.prompt_to_target_agent,
                output={feature.value: r.output},
                source_agents=_flatten_sources(results_list),
            )

    # 2. Full failure.
    if all(r.status == AgentStatus.FAILED for _, r in results_list):
        return AgentResult(
            status=AgentStatus.FAILED,
            output={feature.value: r.output for feature, r in results_list},
            source_agents=_flatten_sources(results_list),
        )

    # 3. Success (possibly partial - some FAILED, some COMPLETED).
    combined_output: dict[str, object] = {}
    map_url = None
    hotspots = None
    migration_route = None
    observation_count = None
    confidences: list[float] = []

    for feature, r in results_list:
        combined_output[feature.value] = r.output
        if map_url is None and r.map_url:
            map_url = r.map_url
        if hotspots is None and r.hotspots:
            hotspots = r.hotspots
        if migration_route is None and r.migration_route:
            migration_route = r.migration_route
        if observation_count is None and r.observation_count is not None:
            observation_count = r.observation_count
        if r.confidence is not None:
            confidences.append(r.confidence)

    # If there is only one worker in play, hand back its raw payload so
    # callers do not have to reach through a dict for a single result.
    output: object
    if len(results_list) == 1:
        output = results_list[0][1].output
    else:
        output = combined_output

    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=output,
        map_url=map_url,
        hotspots=hotspots,
        migration_route=migration_route,
        observation_count=observation_count,
        confidence=(sum(confidences) / len(confidences)) if confidences else None,
        source_agents=_flatten_sources(results_list),
    )


def _flatten_sources(
    results_list: list[tuple[BiodiversityFeature, AgentResult]],
) -> list[str]:
    seen: list[str] = ["Biodiversity Agent Orchestrator"]
    for _, r in results_list:
        for s in r.source_agents:
            if s not in seen:
                seen.append(s)
    return seen
