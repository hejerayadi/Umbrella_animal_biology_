from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ToyState:

    """Custom object passed through the graph — stands in for our real
    TraitDiscoveryInput/Output dataclasses """

    question: str
    answer: str = ""
    needs_escalation: bool = False
    escalation_target: Optional[str] = None
    parallel_result_a: str = ""
    parallel_result_b: str = ""
    merged_result: str = ""