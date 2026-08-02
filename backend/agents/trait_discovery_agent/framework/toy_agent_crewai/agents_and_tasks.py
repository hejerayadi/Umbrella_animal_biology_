import os
from crewai import Agent, Task, LLM
from framework.toy_agent_crewai.state import ToyTaskOutput, MergedOutput

def get_nim_crewai_llm() -> LLM:
    """CrewAI talks to models through LiteLLM's provider strings, not a LangChain client —
    this is a second, separate NIM integration path from the LangGraph one in llm_client.py, and
    is itself a data point for the benchmark (two integrations to maintain vs. one).

    IMPORTANT: LiteLLM's nvidia_nim provider does NOT reliably read credentials from the
    api_key= kwarg passed to crewai's LLM() — it looks for the NVIDIA_NIM_API_KEY environment
    variable specifically (a different name from the NVIDIA_NIM_API_KEY used by ChatNVIDIA in
    llm_client.py). Without it, calls fail with:
        litellm.InternalServerError: Nvidia_nimException - Missing credentials...
    We only ask the user to maintain ONE secret (NVIDIA_NIM_API_KEY) in .env, and mirror it into
    the name LiteLLM expects here, so both integration paths work off a single stored key."""
    api_key = os.environ["NVIDIA_NIM_API_KEY"]
    os.environ.setdefault("NVIDIA_NIM_API_KEY", api_key)  # the var LiteLLM's nvidia_nim provider reads

    return LLM(
        model=f"nvidia_nim/{os.environ.get('NIM_MODEL', 'meta/llama-3.1-70b-instruct')}",
        api_key=api_key,
        temperature=float(os.environ.get("NIM_TEMPERATURE", "0.2")),
        timeout=30,   # covers BOTH the main completion call AND CrewAI's internal
                       # instructor-based structured-output conversion call — without this,
                       # a stalled request on either call hangs indefinitely
    )

def build_responder_agent() -> Agent:
    return Agent(
        role="Responder",
        goal="Answer the question and flag if it can't be answered confidently.",
        backstory="A focused domain-question answerer for the toy benchmark.",
        llm=get_nim_crewai_llm(),
        verbose=True,
    )

def build_responder_task(agent: Agent, question: str) -> Task:
    return Task(
        description=(
            f"Answer this question: {question}\n"
            "If you cannot answer confidently, set needs_escalation=true."
        ),
        expected_output="A structured answer with an escalation flag.",
        agent=agent,
        output_pydantic=ToyTaskOutput,
    )

def build_parallel_agent(name: str) -> Agent:
    return Agent(
        role=f"Parallel worker {name}",
        goal="Process a fragment of the question independently.",
        backstory="One of two workers meant to run concurrently.",
        llm=get_nim_crewai_llm(),
        verbose=True,
    )

def build_parallel_task(agent: Agent, question: str, label: str) -> Task:
    return Task(
        description=f"Summarize this fragment in under 10 words: {question}",
        expected_output=f"A short summary labeled {label}.",
        agent=agent,
        async_execution=True,   # this is CrewAI's parallel-execution primitive
    )

def build_merge_agent() -> Agent:
    return Agent(
        role="Consolidator",
        goal="Combine two independent summaries into one merged result.",
        backstory="Waits for both parallel workers, then reconciles their output.",
        llm=get_nim_crewai_llm(),
        verbose=True,
    )

def build_merge_task(agent: Agent, task_a: Task, task_b: Task) -> Task:
    """This task is REQUIRED, not optional — CrewAI raises a validation error if a crew's
    task list ends with more than one async_execution=True task in a row. Unlike LangGraph,
    which lets you attach a dedicated merge node to two parallel edges implicitly, CrewAI
    needs an explicit synchronous task at the end of the list, wired via `context=[...]`,
    to actually collect and consolidate async task results."""
    return Task(
        description="Combine the two labeled summaries you were given into one merged string.",
        expected_output="A single string containing both summaries, clearly labeled.",
        agent=agent,
        context=[task_a, task_b],   # this is what lets the task see task_a/task_b outputs
        output_pydantic=MergedOutput,
    )