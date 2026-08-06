from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentCard:
    name: str
    description: str
    capabilities: list[str]
    required_inputs: list[str]
    produced_outputs: list[str]
