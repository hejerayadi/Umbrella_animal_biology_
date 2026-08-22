"""Build structured FLUX prompts from organized visualization input."""

from __future__ import annotations

from typing import Any

PROMPT_TEMPLATE = """Create a professional 2D scientific visualization based on the following biological information:

{structured_information}

Requirements:
- scientific illustration style
- clear representation of the described characteristics
- coherent visual relationship between the identified traits
- clean composition
- high visual quality
- no invented biological claims
- no unnecessary text
- no decorative elements unrelated to the described biology
"""


def build_visualization_prompt(
    organized: dict[str, Any],
    instruction: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Compose a single FLUX prompt from organized sections and context."""
    del context  # reserved for future prompt enrichment
    sections = organized.get("visualization_input", {}).get("sections", [])
    structured = _format_sections(sections, instruction)
    return PROMPT_TEMPLATE.format(structured_information=structured.strip())


def _format_sections(sections: list[dict[str, str]], instruction: str) -> str:
    lines: list[str] = []

    if instruction.strip():
        lines.append(f"User request: {instruction.strip()}")

    for section in sections:
        section_type = section.get("type", "information").replace("_", " ").title()
        content = section.get("content", "").strip()
        if content:
            lines.append(f"- {section_type}: {content}")

    return "\n".join(lines)
