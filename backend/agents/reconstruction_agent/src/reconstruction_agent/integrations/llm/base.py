"""What the agent needs from a language model, and nothing more.

The graph asks a model for exactly two things: which action to take next, and
what is wrong with the evidence it has. Both are choices from a closed set, so
both come back as a validated Pydantic model rather than as prose.

This narrow protocol is what lets `LLM_PROVIDER=none` be a first-class mode
instead of a degraded one. The deterministic client below satisfies the same
interface, so the graph has one code path whether or not a key is configured -
and the tests exercise the same path the production run takes.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

#: The structured reply a call is expected to produce.
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class LLMClient(Protocol):
    """A model that answers with a validated object, or declines to answer."""

    @property
    def available(self) -> bool:
        """Whether calling this client can produce a model-authored answer.

        False for the null client. Callers branch on this rather than catching
        an exception, because "no model configured" is an expected mode of
        operation and not an error condition.
        """
        ...

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[StructuredT],
    ) -> StructuredT | None:
        """One structured completion, or None when the model gave nothing usable.

        Returning None rather than raising is deliberate: a model that answers
        with unparseable output is a reason to fall back to the deterministic
        strategy, not a reason to fail a run that has real evidence in hand.
        """
        ...
