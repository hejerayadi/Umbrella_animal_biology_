from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class JournalTopic:
    name: str
    subfield: Optional[str] = None
    field: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class Journal:
    id: str
    name: str
    publisher: Optional[str]
    issn: Optional[str]

    works_count: int
    cited_by_count: int

    country: Optional[str]
    homepage: Optional[str]

    # Publication constraints users express in prose ("open access only",
    # "nothing with a big APC"). Kept out of the embedding text on purpose:
    # they are filtering facts, not part of a journal's research scope.
    is_oa: bool = False
    is_in_doaj: bool = False
    apc_usd: Optional[int] = None

    topics: List[JournalTopic] = field(default_factory=list)