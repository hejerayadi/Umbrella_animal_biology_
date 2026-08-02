from pydantic import BaseModel

class ToyTaskOutput(BaseModel):
    """Structured output model — CrewAI's answer to 'pass a custom object, not a raw string.'
    Tasks can declare output_pydantic=ToyTaskOutput so results deserialize into this, rather than
    only ever returning free text."""
    answer: str
    needs_escalation: bool

class MergedOutput(BaseModel):
    """Structured output for the explicit consolidation task that follows the two async
    parallel tasks — see the note in crew.py about why this task has to exist at all."""
    merged_summary: str