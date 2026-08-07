from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from app.contracts.agent_task import AgentTask
from app.domain.enums import AnalysisStatus
from app.persistence.db import Database
from app.persistence.models import AgentRun, OrchestratorTask


class AnalysisRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, task: AgentTask) -> tuple[UUID, bool]:
        with self.database.session() as session:
            existing = session.scalar(
                select(OrchestratorTask).where(OrchestratorTask.idempotency_key == task.idempotency_key)
            )
            if existing:
                return UUID(existing.analysis_id), False
            analysis_id = uuid4()
            session.add(
                OrchestratorTask(
                    analysis_id=str(analysis_id),
                    task_id=str(task.task_id),
                    trace_id=str(task.trace_id),
                    idempotency_key=task.idempotency_key,
                    status=AnalysisStatus.received.value,
                    request_json=task.model_dump(mode="json"),
                )
            )
            session.commit()
            return analysis_id, True

    def update(
        self,
        analysis_id: UUID,
        status: AnalysisStatus,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.database.session() as session:
            task = session.get(OrchestratorTask, str(analysis_id))
            if task:
                task.status = status.value
                task.response_json = response
                task.error = error
                session.commit()

    def add_run(
        self, analysis_id: UUID, capability: str, status: str, duration_ms: int, error: str | None = None
    ) -> None:
        with self.database.session() as session:
            session.add(
                AgentRun(
                    analysis_id=str(analysis_id),
                    capability=capability,
                    status=status,
                    duration_ms=duration_ms,
                    error=error,
                )
            )
            session.commit()

    def get(self, analysis_id: UUID) -> OrchestratorTask | None:
        with self.database.session() as session:
            record = session.get(OrchestratorTask, str(analysis_id))
            session.expunge(record) if record else None
            return record
