import asyncio
from framework.toy_agent_langgraph.graph import build_toy_graph
from framework.toy_agent_langgraph.state import ToyState

async def main():
    app = build_toy_graph()
    result = await app.ainvoke(ToyState(question="What pathways involve UCP1?"))
    print(result)

if __name__ == "__main__":
    asyncio.run(main())