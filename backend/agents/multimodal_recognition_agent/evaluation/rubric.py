"""The documented human rubric for relevance, explanation quality and task
completion where judgement is genuinely required.

Sprint 4 offers two options for these three criteria: a rubric-based LLM judge,
or a documented human rubric with recorded scores. **This benchmark uses the
human rubric**, deliberately. Adding a judge model in Phase 2 would introduce a
second, unvalidated model into the measurement path, and its calls would have to
be tracked separately from the agent's two-call ceiling to keep that ceiling
verifiable. A written rubric with recorded scores costs a reviewer some time and
costs the evaluation nothing in trustworthiness.

**A human score never overrides a deterministic verdict.** If a reviewer thinks
an explanation is excellent and the agent named the wrong species, the species
metric still fails. The rubric measures how well the agent communicated, not
whether it was right.

Scores are integers 1-5. A reviewer records them into the `human_review` block
of a case result; unscored cases stay `scored: false` and are reported as
unscored rather than as zero.
"""
from __future__ import annotations

from typing import Any

SCALE_MIN, SCALE_MAX = 1, 5

RELEVANCE = {
    "criterion": "relevance_to_the_instruction",
    "question": "Does the response answer what the user actually asked about this image?",
    "levels": {
        5: "Answers the instruction directly and completely, including any part the "
           "agent cannot provide, which it declines out loud rather than ignoring.",
        4: "Answers the instruction; a secondary part of the question is addressed only "
           "in passing.",
        3: "Answers the general subject but not the specific question asked.",
        2: "Largely off-target: responds to a question the user did not ask.",
        1: "Unrelated to the instruction.",
    },
}

EXPLANATION_QUALITY = {
    "criterion": "explanation_quality",
    "question": "Is the explanation accurate, grounded in the evidence actually "
                "produced, and honest about its own limits?",
    "levels": {
        5: "Grounded entirely in the returned candidates and decision; states "
           "uncertainty where the decision was uncertain; claims no probability.",
        4: "Grounded and honest, but vague or repetitive.",
        3: "Broadly correct, with one unsupported or confusing statement.",
        2: "Names evidence the run did not produce, or overstates confidence.",
        1: "Asserts a probability, invents a taxon, or contradicts the decision.",
    },
    "hard_failures": [
        "any species named that is not among the returned candidates",
        "any phrasing that presents a ranking score as a probability or percentage",
        "any claim that taxonomy was verified live when provenance says mock",
    ],
}

TASK_COMPLETION_JUDGEMENT = {
    "criterion": "task_completion_human",
    "question": "Would a working biologist consider this a usable answer to the "
                "request, given what the image actually contained?",
    "levels": {
        5: "Usable as-is, including when the honest answer was 'not identified'.",
        4: "Usable, but the reader must do a little work to act on it.",
        3: "Partially usable; an important part of the request is unaddressed.",
        2: "Not usable without re-running or asking again.",
        1: "Misleading - it would send the reader in the wrong direction.",
    },
    "note": "Scored independently of the deterministic `task_completion` metric, "
            "which checks the terminal state and payload rather than usefulness.",
}

CRITERIA = (RELEVANCE, EXPLANATION_QUALITY, TASK_COMPLETION_JUDGEMENT)

PROTOCOL = """\
1. Score from the sanitized result file, not from the agent's internals.
2. Score relevance and explanation quality before looking at whether the species
   was correct, so a right answer does not flatter a poor explanation.
3. Record the reviewer's name and any note explaining a score below 4.
4. Leave a case unscored rather than guessing; unscored is reported as unscored.
5. Never edit a deterministic metric because of a human score.
"""


def validate_score(value: Any) -> bool:
    return isinstance(value, int) and SCALE_MIN <= value <= SCALE_MAX


def describe() -> dict[str, Any]:
    """The rubric as data, so the report can print it without restating it."""
    return {
        "method": "documented_human_rubric",
        "llm_judge_used": False,
        "scale": {"min": SCALE_MIN, "max": SCALE_MAX},
        "criteria": [
            {k: v for k, v in criterion.items() if k != "levels"}
            | {"levels": {str(k): v for k, v in criterion["levels"].items()}}
            for criterion in CRITERIA
        ],
        "protocol": PROTOCOL,
        "overrides_deterministic_metrics": False,
    }
