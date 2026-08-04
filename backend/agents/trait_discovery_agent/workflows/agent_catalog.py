import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

_AGENT_CARDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_cards")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class AgentCard:
    name: str
    kind: str  # "internal" or "external"
    body: str


def _render_json_body(data: dict) -> str:
    parts: list[str] = []

    role = data.get("role") or data.get("description") or data.get("body")
    if role:
        parts.append(f"Role: {role}")

    if data.get("call_when"):
        parts.append(f"Call when: {data['call_when']}")

    if data.get("cannot_help_with"):
        cannot_help = ", ".join(data["cannot_help_with"])
        parts.append(f"Cannot help with: {cannot_help}")

    if data.get("returns"):
        parts.append(f"Returns: {data['returns']}")

    return "\n".join(parts)


def _parse_card(path: str) -> AgentCard:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    if path.endswith(".json"):
        data = json.loads(text)
        return AgentCard(
            name=data.get("name", os.path.splitext(os.path.basename(path))[0]),
            kind=data.get("kind", "internal"),
            body=_render_json_body(data),
        )

    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{path} is missing YAML frontmatter (---\nname: ...\nkind: ...\n---)")
    frontmatter, body = match.groups()
    fields = dict(line.split(":", 1) for line in frontmatter.strip().splitlines())
    return AgentCard(name=fields["name"].strip(), kind=fields.get("kind", "internal").strip(), body=body.strip())


@lru_cache(maxsize=1)
def load_agent_cards() -> List[AgentCard]:
    return [
        _parse_card(os.path.join(_AGENT_CARDS_DIR, fname))
        for fname in sorted(os.listdir(_AGENT_CARDS_DIR))
        if fname.endswith((".md", ".json"))
    ]


def build_catalog_text(exclude: Optional[str] = None) -> str:
    """Renders every card except `exclude` — the resolver's own rule is
    'never pick the waiting agent itself,' so we don't even show it the option."""
    cards = [c for c in load_agent_cards() if c.name != exclude]

    def render_card(card: AgentCard) -> str:
        if card.body:
            return f"### {card.name}\n\n{card.body}"
        return f"### {card.name}"

    return "\n\n".join(render_card(card) for card in cards)