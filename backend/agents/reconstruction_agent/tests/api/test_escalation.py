"""Escalation toward the Genome Agent, and the loop guard it has to satisfy.

`card.json` says this agent works on a specific sequence and does not pick one.
A message naming only a species is therefore under-specified rather than
malformed - another agent can supply what is missing, and asking is a hand-off,
not a failure.

The orchestrator makes that hand-off conditional. `worker_node.py` snapshots
`sorted(state.context)` on every escalation and force-fails a second one whose
signature is unchanged, because that guard is what stopped a single unmet
dependency from becoming 36 steps of external calls. An escalation carrying no
new context key is indistinguishable from a loop.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from tests.fakes import isolated_settings

from reconstruction_agent.api.dependencies import get_reconstruction_service
from reconstruction_agent.main import create_app
from reconstruction_agent.services.reconstruction.escalation import (
    FINDINGS_KEY,
    GENOME_AGENT,
    build_escalation,
    should_escalate,
)


class _NeverCalled:
    """The service must not run: escalation happens before any work."""

    def __init__(self) -> None:
        self.calls = 0

    async def reconstruct(self, request: Any) -> Any:
        self.calls += 1
        raise AssertionError("the service ran on an under-specified request")


def _client(service: _NeverCalled) -> TestClient:
    app = create_app(isolated_settings())
    app.dependency_overrides[get_reconstruction_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def _execute(client: TestClient, instruction: str, context: dict[str, Any]) -> Any:
    return client.post(
        "/execute",
        headers={"X-Trace-Id": "trace-escalation", "X-Request-Id": "req-1"},
        json={"instruction": instruction, "context": context},
    )


class TestWhenEscalationIsTheRightAnswer:
    def test_a_species_with_no_sequence_escalates(self) -> None:
        assert should_escalate({"scientific_name": "Ursus maritimus"})

    def test_a_named_sequence_never_escalates(self) -> None:
        """There is nothing to ask for: the target is already specified."""
        assert not should_escalate(
            {"scientific_name": "Ursus maritimus", "sequence_accession": "NC_003428.1"}
        )

    def test_a_pasted_sequence_never_escalates(self) -> None:
        assert not should_escalate({"species": "Ursus maritimus", "sequence": "ACGTNNNNACGT"})

    def test_a_request_naming_neither_does_not_escalate(self) -> None:
        """Nobody can supply what this lacks, so escalating spends a step to be
        told the same thing."""
        assert not should_escalate({})
        assert not should_escalate({"assembly_level": "Scaffold"})


class TestTheEscalationPayload:
    def test_it_names_what_is_needed_and_from_whom(self) -> None:
        prompt, findings = build_escalation(
            "Fill in the polar bear genome.",
            {"scientific_name": "Ursus maritimus"},
            "no accession supplied",
        )

        assert "Ursus maritimus" in prompt
        assert "accession" in prompt
        assert findings[FINDINGS_KEY]["needs"] == ["sequence_accession", "target_gaps"]

    def test_it_carries_what_was_already_known(self) -> None:
        """So the Genome Agent does not re-derive what this agent was given."""
        _, findings = build_escalation(
            "Fill the gaps.",
            {"species": "Ursus maritimus", "assembly_id": "GCF_000687225.1"},
            "no accession",
        )

        assert findings[FINDINGS_KEY]["known"]["assembly_id"] == "GCF_000687225.1"

    def test_the_findings_key_is_one_the_genome_agent_does_not_set(self) -> None:
        """The whole point: the context signature must change, or the
        orchestrator reads the escalation as a loop and fails the run."""
        genome_agent_keys = {
            "genome",
            "assembly_id",
            "gene_list",
            "gene_table",
            "genome_metadata",
            "species_record",
            "visualization",
            "explanation",
            "scientific_name",
            "sequence_accession",
            "assembly_level",
            "target_gaps",
        }
        assert FINDINGS_KEY not in genome_agent_keys


class TestTheEscalationOverHttp:
    def test_it_answers_needs_agent_without_doing_any_work(self) -> None:
        service = _NeverCalled()
        response = _execute(
            _client(service),
            "Fill in the gaps in the polar bear genome.",
            {"scientific_name": "Ursus maritimus"},
        )
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "needs_agent"
        assert body["target_agent"] == GENOME_AGENT
        assert body["prompt_to_target_agent"]
        assert service.calls == 0

    def test_the_output_adds_a_context_key(self) -> None:
        response = _execute(
            _client(_NeverCalled()),
            "Reconstruct the polar bear genome.",
            {"species": "Ursus maritimus"},
        )
        output = response.json()["output"]

        assert isinstance(output, dict), "only a dict is merged into the shared context"
        assert FINDINGS_KEY in output

    def test_a_request_nobody_can_help_with_still_fails_cleanly(self) -> None:
        response = _execute(_client(_NeverCalled()), "Reconstruct something.", {})
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "failed"
        assert body["error"]
