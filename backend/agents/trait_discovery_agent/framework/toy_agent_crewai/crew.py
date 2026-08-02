from crewai import Crew, Process
from framework.toy_agent_crewai.agents_and_tasks import (
    build_responder_agent, build_responder_task,
    build_parallel_agent, build_parallel_task,
    build_merge_agent, build_merge_task,
)

def run_toy_crew(question: str):
    responder = build_responder_agent()
    responder_task = build_responder_task(responder, question)

    worker_a, worker_b = build_parallel_agent("A"), build_parallel_agent("B")
    task_a = build_parallel_task(worker_a, question, "A")
    task_b = build_parallel_task(worker_b, question, "B")

    merger = build_merge_agent()
    merge_task = build_merge_task(merger, task_a, task_b)

    crew = Crew(
        agents=[responder, worker_a, worker_b, merger],
        # order matters: task_a/task_b (async) must NOT be the last items in the list —
        # merge_task (sync) has to trail them so the crew has something synchronous to
        # finish on. This is the fix for the "must end with at most one asynchronous
        # task" ValidationError.
        tasks=[responder_task, task_a, task_b, merge_task],
        process=Process.sequential,
        verbose=True,
    )
    crew_output = crew.kickoff()

    structured = responder_task.output.pydantic     # ToyTaskOutput instance
    merged = merge_task.output.pydantic              # MergedOutput instance, built BY the framework
                                                       # via context=[task_a, task_b], not by us
                                                       # stitching strings together in Python

    # conditional branch, still done in plain Python — CrewAI has no add_conditional_edges equivalent
    escalation_target = "human_reviewer" if structured.needs_escalation else None

    return {
        "answer": structured.answer,
        "needs_escalation": structured.needs_escalation,
        "escalation_target": escalation_target,
        "merged_result": merged.merged_summary,
    }