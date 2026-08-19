from pydantic import BaseModel
from typing import Literal


class DocPayload(BaseModel):
    doc_id: str
    source_dataset: Literal["arxiv", "pubmed", "multi_xscience", "scireviewgen"]
    domain: str
    section_type: Literal["abstract", "body", "related_work", "review"]
    input_text: str
    target_text: str
    n_tokens: int
    language: str = "en"


class CitationPayload(BaseModel):
    doc_id: str
    source_dataset: Literal["scicite"] = "scicite"
    citing_paper_id: str
    cited_paper_id: str
    intent: Literal["Background", "Method", "Result Comparison"]
    section_name: str
    context: str