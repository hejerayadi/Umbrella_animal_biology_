from __future__ import annotations

from fastapi import FastAPI

from .logic import ImageGenerationLogic
from .schema import AgentRequest, AgentResult, AgentStatus

app = FastAPI(
    title="Image Generation Agent",
    description=(
        "Routes to Trait Discovery when biological traits are missing, then "
        "generates a 2D scientific illustration with FLUX.2-pro."
    ),
    version="1.0.0",
)

_logic = ImageGenerationLogic()


@app.post("/execute", response_model=AgentResult)
def execute(request: AgentRequest) -> AgentResult:
    try:
        return _logic.run(request)
    except Exception as exc:
        return AgentResult(status=AgentStatus.FAILED, output=str(exc))
