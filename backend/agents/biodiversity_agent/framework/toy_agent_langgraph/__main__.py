"""Run the toy LangGraph agent.

    python -m backend.agents.biodiversity_agent.framework.toy_agent_langgraph
"""

from __future__ import annotations

import asyncio

from .graph import build_toy_graph
from .state import ToyState


async def main() -> None:
    app = build_toy_graph()
    result = await app.ainvoke(ToyState(question="What pathways involve UCP1?"))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
