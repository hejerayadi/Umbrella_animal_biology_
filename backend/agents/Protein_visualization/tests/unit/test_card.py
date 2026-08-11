import json
from pathlib import Path


def test_platform_card_declares_the_prose_structure_contract() -> None:
    agent_root = Path(__file__).resolve().parents[2]
    card = json.loads((agent_root / "card.json").read_text(encoding="utf-8"))

    assert card["output"]["protein_structure"] == "string"
