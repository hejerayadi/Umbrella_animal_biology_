from __future__ import annotations

from .orchestrator import aggregate_results, decide_after_discovery, decide_next_after_routing


def test_decide_next_after_routing_matches_the_four_cases() -> None:
    assert decide_next_after_routing({"route": "discovery"}) == ["discovery"]
    assert decide_next_after_routing({"route": "writing"}) == ["writing"]
    assert decide_next_after_routing({"route": "both_sequential"}) == ["discovery"]
    assert decide_next_after_routing({"route": "both_parallel"}) == ["discovery", "writing"]
    assert decide_next_after_routing({"route": "unknown"}) == ["discovery"]


def test_decide_after_discovery_routes_only_sequential_to_writing() -> None:
    assert decide_after_discovery({"route": "both_sequential"}) == "writing"
    assert decide_after_discovery({"route": "both_parallel"}) == "aggregate"
    assert decide_after_discovery({"route": "discovery"}) == "aggregate"


def test_aggregate_results_always_returns_discovery_and_writing_keys() -> None:
    result = aggregate_results({
        "route": "both_parallel",
        "discovery_result": None,
        "writing_result": None,
    })

    assert result["final_result"].output == {"discovery": None, "writing": None}
