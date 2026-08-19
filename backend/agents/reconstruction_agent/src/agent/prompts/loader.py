"""Loads prompts from markdown files in `src/prompts/`.

Prompts are data, not code. Keeping them as `.md` means they can be reviewed,
diffed and edited by someone who does not read Python - and a prompt change
alters agent behaviour as much as a code change does, so it deserves to be a
visible diff on its own file rather than a buried string edit.

Templates use `str.format` placeholders (`{instruction}`). Any literal brace -
the JSON examples in the prompts are full of them - must be doubled in the
template; `render` reports a bad placeholder by name rather than raising a
bare KeyError.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from configuration.settings import PROMPTS_DIR


class PromptNotFoundError(FileNotFoundError):
    """A prompt file named in code is missing from `src/prompts/`."""


class PromptRenderError(ValueError):
    """A template placeholder had no matching value."""


@lru_cache(maxsize=64)
def load(name: str, *, prompts_dir: Path | None = None) -> str:
    """The raw text of one prompt, by stem (e.g. `"planner.system"`).

    Cached: prompts do not change within a process, and re-reading them per
    LLM call would put disk I/O in the request path.
    """
    directory = prompts_dir or PROMPTS_DIR
    path = directory / f"{name}.md"

    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.md")) if directory.is_dir() else []
        raise PromptNotFoundError(
            f"No prompt '{name}.md' in {directory}. Available: {', '.join(available) or 'none'}"
        )

    return path.read_text(encoding="utf-8").strip()


def render(name: str, /, **values: object) -> str:
    """Load a prompt and substitute its placeholders.

    Raises `PromptRenderError` naming the missing key, because the alternative
    - a KeyError from deep inside `str.format` - gives no clue which prompt or
    which placeholder was at fault.
    """
    template = load(name)
    try:
        return template.format(**values)
    except KeyError as error:
        raise PromptRenderError(
            f"Prompt '{name}' needs a value for {error} that was not supplied."
        ) from error
    except IndexError as error:
        raise PromptRenderError(
            f"Prompt '{name}' contains an unescaped brace; literal braces must be doubled."
        ) from error


def available(prompts_dir: Path | None = None) -> list[str]:
    """Every prompt stem on disk. Used by the tests to keep files and code in step."""
    directory = prompts_dir or PROMPTS_DIR
    return sorted(path.stem for path in directory.glob("*.md")) if directory.is_dir() else []


def clear_cache() -> None:
    """Forget loaded prompts, so an edit takes effect without a restart."""
    load.cache_clear()
