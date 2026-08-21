# Framework Benchmark Results — Trait Discovery Agent

| Criterion | LangGraph | CrewAI | Notes |
|---|---|---|---|
| Custom object / state passing | Pass | Partial | LangGraph: typed dataclass flows through every node untouched. CrewAI: needs `output_pydantic` per task, and cross-task state isn't a single shared object — you read it back off each Task's `.output` after the fact. |
| Parallel + sequential execution | Pass | Partial | LangGraph: native fan-out/fan-in edges, single merge node. CrewAI: `async_execution=True` runs tasks concurrently but there's no native "merge" node — you assemble the merge yourself in plain Python after `kickoff()`. |
| Conditional / branching execution | Pass | Fail | LangGraph: `add_conditional_edges` is a first-class graph primitive. CrewAI: no equivalent — branching has to happen outside the crew, in the calling code, which means `NEEDS_AGENT`-style routing isn't expressible *inside* CrewAI's own execution model. |
| LLM/tool-calling integration (NIM) | Pass | Pass | Both reach NVIDIA NIM successfully. LangGraph via `langchain-nvidia-ai-endpoints` (ChatNVIDIA). CrewAI via LiteLLM's `nvidia_nim/` provider string — a second, separate integration surface to maintain. |
| Observability / debugging | Pass | Partial | LangGraph: `astream` exposes intermediate node state as it runs. CrewAI: `verbose=True` gives readable agent/task logs but not the same structured intermediate-state introspection. |
| Async support | Pass | Partial | LangGraph nodes are `async def` throughout. CrewAI's `async_execution=True` parallelizes tasks but the top-level `kickoff()` call itself is synchronous. |
| Learning curve | Partial | Pass | CrewAI's role/Task/Crew vocabulary got a working toy agent up faster. LangGraph took longer to get right, mainly around wiring conditional edges correctly. |
| Community/maturity | Pass | Pass | Both actively maintained with recent commits; docs quality comparable for this use case. |

**Decision: LangGraph.**
CrewAI prototyped faster, but failed the criterion that matters most for this project — conditional
branching as a first-class part of the framework's own execution model. Since `NEEDS_AGENT` routing is
central to every orchestrator and sub-orchestrator in this agent, needing to hand-roll that branching
outside the framework (as §6.3 does) defeats the point of adopting a framework for orchestration in the
first place. LangGraph's slightly steeper learning curve is a one-time cost; CrewAI's missing branching
primitive would be a recurring one, on every escalation path in the real agent.