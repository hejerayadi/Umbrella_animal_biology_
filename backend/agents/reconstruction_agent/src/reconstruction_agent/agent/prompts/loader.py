"""Loading and rendering the agent's Markdown prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

_SYSTEM_HEADING = "# System\n"
_USER_HEADING = "\n# User template\n"
_AVAILABLE_PROMPTS = frozenset({"planner", "critic"})


@dataclass(frozen=True)
class PromptTemplate:
    """A model system message and a formatted user-message template."""

    system: str
    user_template: str

    def render_user(self, **values: Any) -> str:
        """Render the user template with the typed values supplied by the node."""
        return self.user_template.format(**values)


@cache
def load_prompt(name: str) -> PromptTemplate:
    """Read one packaged Markdown prompt once, then reuse it for the process.

    Keeping the Markdown beside this module makes the source tree readable and
    works both from an editable ``uv`` environment and from a built wheel.
    """
    if name not in _AVAILABLE_PROMPTS:
        raise ValueError(f"Unknown prompt {name!r}.")

    content = (
        files("reconstruction_agent.agent.prompts")
        .joinpath(f"{name}.md")
        .read_text(encoding="utf-8")
    )
    if not content.startswith(_SYSTEM_HEADING) or _USER_HEADING not in content:
        raise ValueError(f"Prompt {name!r} must contain '# System' followed by '# User template'.")

    system, user_template = content[len(_SYSTEM_HEADING) :].split(_USER_HEADING, maxsplit=1)
    return PromptTemplate(system=system.strip(), user_template=user_template.strip())
