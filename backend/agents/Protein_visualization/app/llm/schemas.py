from typing import Literal

from pydantic import BaseModel, Field


class ExplanationOutput(BaseModel):
    """Structured explanation. Every sentence must be traceable to a supplied fact."""

    summary: str = Field(description="Two to four sentences grounded strictly in the supplied facts.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Restatement of the supplied limitations; never invent new ones.",
    )


class CriticOutput(BaseModel):
    """Scientific critic verdict. May only confirm or downgrade the deterministic verdict."""

    verdict: Literal["ACCEPT", "REVISE", "ABSTAIN"]
    reasons: list[str] = Field(default_factory=list)
