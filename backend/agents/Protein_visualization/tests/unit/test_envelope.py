from backend.agents.Protein_visualization.app.contracts.envelope import (
    ApiResponse,
    ErrorCode,
    failure,
    success,
)
from backend.agents.Protein_visualization.app.observability.context import log_context


def test_success_carries_correlation_ids_from_the_log_context() -> None:
    with log_context(request_id="req-1", trace_id="trace-1", task_id="task-1"):
        response: ApiResponse[dict[str, str]] = success({"accession": "P04637"})

    assert response.data == {"accession": "P04637"}
    assert response.error is None
    assert response.meta.request_id == "req-1"
    assert response.meta.trace_id == "trace-1"
    assert response.meta.task_id == "task-1"


def test_failure_leaves_data_null_and_mirrors_the_trace_id() -> None:
    with log_context(trace_id="trace-2"):
        response = failure(
            code=ErrorCode.protein_not_found,
            title="Protein not found",
            status=404,
            detail="No canonical accession",
            instance="/api/v1/protein-structure-analyses",
        )

    assert response.data is None
    assert response.error is not None
    assert response.error.code == "PROTEIN_NOT_FOUND"
    assert response.error.status == 404
    assert response.error.trace_id == "trace-2"
    assert response.meta.trace_id == "trace-2"


def test_envelope_serializes_all_three_keys() -> None:
    body = success({"ok": True}).model_dump(mode="json")
    assert set(body) == {"data", "meta", "error"}
