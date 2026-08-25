"""Example test to demonstrate Literature Agent functionality end-to-end.

These call the real Azure deployments and assert the run completed, so they
need live credentials and fail offline for reasons unrelated to the code.
`pytest.ini` deselects them by default; run them with:

    pytest -m integration
"""

import pytest

from ..api import execute
from ..schema import AgentRequest

pytestmark = pytest.mark.integration


def test_discovery_only():
    """Test: User asks for knowledge discovery only."""
    request = AgentRequest(
        instruction="Find recent papers on animal behavior genetics",
        context={"source": "user_query"}
    )
    result = execute(request)
    assert result.status.value == "completed"
    assert result.output.get("discovery") is not None
    print("\n✅ Discovery-only route:")
    print(f"   Status: {result.status.value}")
    print(f"   Discovery output: {result.output.get('discovery')}")
    print()


def test_writing_only():
    """Test: User asks for scientific writing only."""
    request = AgentRequest(
        instruction="Write an abstract about genetic variation in species",
        context={"source": "user_query"}
    )
    result = execute(request)
    assert result.status.value == "completed"
    assert result.output.get("writing") is not None
    print("✅ Writing-only route:")
    print(f"   Status: {result.status.value}")
    print(f"   Writing output: {result.output.get('writing')}")
    print()


def test_both_sequential():
    """Test: User asks for both discovery and writing sequentially."""
    request = AgentRequest(
        instruction="Find papers on biodiversity and then write a comprehensive review",
        context={"source": "user_query"}
    )
    result = execute(request)
    assert result.status.value == "completed"
    print("✅ Both (sequential) route:")
    print(f"   Status: {result.status.value}")
    print(f"   Discovery + Writing executed")
    print(f"   Output keys: {list(result.output.keys())}")
    print()


def test_publication_support():
    """Test: User asks for publication/journal recommendations."""
    request = AgentRequest(
        instruction="Recommend journals for publishing our biodiversity study and help draft a paper",
        context={"source": "user_query"}
    )
    result = execute(request)
    assert result.status.value == "completed"
    print("✅ Publication support route:")
    print(f"   Status: {result.status.value}")
    print(f"   Output: {result.output}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING LITERATURE AGENT WITH REAL EXAMPLES")
    print("=" * 70)
    
    test_discovery_only()
    test_writing_only()
    test_both_sequential()
    test_publication_support()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
