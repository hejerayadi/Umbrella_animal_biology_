"""LLM-driven entity extractor: turns the user's sentence into shared facts.

Every worker agent reads what it needs out of `context` - the Genome and
Biodiversity agents both start with `context["species"]` - but nothing was
ever putting the subject of the question in there. `context` starts empty and
only ever collects agent *outputs*, so on a text-only question
`context["species"]` was never set: the species name sat unread in
`instruction` while those agents failed with "Species not specified." before
doing any work.

This step closes that gap. It runs once, between the planner and the first
agent, and seeds `context` with the entities named in the user's message, so
the assumption every agent was already written against becomes true. No agent
code changes.

IMPORTANT - only ever seed keys that describe the QUESTION, never keys an
agent produces as OUTPUT. Agents read the presence of an output key as "that
work is already done": `EvolutionMock` skips the Genome agent once it sees
`context["genome"]`, and the Trait agent is skipped the same way by anyone who
sees `context["traits"]`. Seeding either would make an agent silently skip a
real dependency and answer from nothing. That is why the trait and gene keys
below are named `trait_name` and `gene_name` - to stay clear of `traits`, the
Trait agent's output key.

`gene_name` is what the Protein Visualization Agent resolves against UniProt,
so its spelling matters more than most: it is read as a gene symbol.

`species` is safe despite also being the Multimodal agent's output: no agent
uses its presence as a "Multimodal already ran" flag, they all read it as
plain input, and a species named in text means the same thing as a species
recognised from a photo.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import get_llm

_logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract the subject of a question for Umbrella, a scientific research platform "
    "for animal biology.\n\n"
    "Read the user's message and pull out only what it actually names. Leave a field "
    "empty when the message does not mention it - never guess, never infer a species "
    "from a genus or a habitat, and never fill a field just to have it filled.\n\n"
    "species: the single animal species the question is about. Prefer the scientific "
    "binomial name when the user gives one, otherwise use the common name exactly as "
    "written. Leave empty if the message names no species, or names several with no "
    "clear main subject.\n\n"
    "trait_name: an observable biological characteristic OF THE ANIMAL ITSELF - "
    "something it has or does, such as 'fur growth', 'cold adaptation', 'body size' or "
    "'tusk length'. This is not the topic of the question: 'habitat', 'conservation "
    "status', 'geographic distribution', 'genome' and 'literature' describe what the "
    "user wants to know, not a trait, so leave the field empty for those. Leave it "
    "empty whenever the message names no specific characteristic.\n\n"
    "gene_name: the gene symbol named in the message (for example 'FGF5', 'UCP1'). "
    "Leave empty if no gene is named.\n\n"
    "mutation: a point mutation explicitly written in the message, using amino-acid "
    "notation such as 'R273H'. Leave empty if none is written.\n\n"
    "residue_position: a single positive residue number explicitly requested. Do not "
    "copy the number out of mutation; leave this empty unless the position is requested "
    "separately.\n\n"
    "accession: a nucleotide database accession explicitly written in the message, such "
    "as 'NC_007596', 'MK123456.1' or 'GCF_000001405.40'. It identifies a sequence "
    "record, so it is never a gene symbol and never a species name. Leave it empty "
    "unless the message actually contains one.\n\n"
    "requested_regions: protein regions explicitly requested, preserving the wording "
    "used by the caller. Return an empty list when none are named.\n\n"
    "preferred_source: set to 'pdb' only for an explicit experimental/PDB preference, "
    "to 'alphafold' only for an explicit predicted/AlphaFold preference, and to 'auto' "
    "only when the caller explicitly asks for the best available source. Otherwise leave "
    "it empty.\n\n"
    "include_explanation: set to true when the caller explicitly asks for an explanation "
    "and false when they explicitly ask for structure/visualization only. Otherwise leave "
    "it empty."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{user_query}"),
    ]
)

# A pasted nucleotide sequence is pulled out here rather than by the model.
# Two reasons: the model would have to echo every base back to us, which is
# slow, expensive and the one kind of field an LLM can silently corrupt; and
# the answer is exactly determinable, so there is nothing to reason about.
#
# The IUPAC nucleotide alphabet. `N` is the one that matters most here - a run
# of them is precisely what the Reconstruction agent looks for.
_IUPAC_NUCLEOTIDES = "ACGTURYKMSWBDHVN"

# 20 bases is short for a real sequence and far longer than any English word
# confined to this alphabet, so the run cannot be prose. Matching is done on
# whitespace-free runs only; wrapped FASTA is handled by joining its lines
# first, below.
_SEQUENCE_RUN = re.compile(rf"[{_IUPAC_NUCLEOTIDES}]{{20,}}", re.IGNORECASE)

# A FASTA record: a '>' description line, then the residues on the lines under
# it. Detected separately because those lines are individually short.
_FASTA_HEADER = re.compile(r"^\s*>.*$", re.MULTILINE)

# The four bases plus the unknown. The full IUPAC set above also contains H, K,
# S, W, B, D, V, R, Y and M, which between them spell ordinary English words -
# "THANKS" is entirely IUPAC letters, and was being swallowed as residues off
# the end of a pasted FASTA record. Real sequence is overwhelmingly ACGTN even
# when it carries ambiguity codes, so requiring that dominance separates a
# residue line from a word without rejecting genuine ambiguity.
_CORE_BASES = frozenset("ACGTN")
_MIN_CORE_SHARE = 0.9


def _looks_like_residues(text: str) -> bool:
    """Whether a run of IUPAC-alphabet characters is really sequence."""
    if not text:
        return False
    upper = text.upper()
    if set(upper) - set(_IUPAC_NUCLEOTIDES):
        return False
    core = sum(1 for character in upper if character in _CORE_BASES)
    return core / len(upper) >= _MIN_CORE_SHARE


def _find_sequence(user_query: str) -> str | None:
    """The nucleotide sequence pasted into the message, if there is one.

    Returns the residues with all whitespace removed, upper-cased, or None.
    Deliberately conservative: a false positive would send the Reconstruction
    agent off to repair a fragment of the user's own sentence.
    """

    # A FASTA block first: its residue lines are wrapped, so each one on its
    # own may be under the length floor even though the record is not.
    for header in _FASTA_HEADER.finditer(user_query):
        residues: list[str] = []
        for line in user_query[header.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                # The newline ending the header itself produces one of these
                # before any residue line is seen, so a blank only terminates
                # the record once it is actually under way.
                if residues:
                    break
                continue
            if stripped.startswith(">") or not _looks_like_residues(stripped):
                break
            residues.append(stripped.upper())
        joined = "".join(residues)
        if len(joined) >= 20:
            return joined

    for match in _SEQUENCE_RUN.finditer(user_query):
        candidate = match.group(0).upper()
        if _looks_like_residues(candidate):
            return candidate

    return None


class _ExtractorOutput(BaseModel):
    """Structured output requested from the LLM.

    Every field is optional: most questions name a species but no gene, or a
    gene but no species. An empty field means "not mentioned", and the
    extractor drops it rather than writing a blank into the shared context.
    """

    species: str | None = Field(
        default=None,
        description="Scientific or common name of the species the question is about. "
        "Empty when the message names no species.",
    )
    trait_name: str | None = Field(
        default=None,
        description="The observable trait or characteristic being asked about. "
        "Empty when the message names no specific trait.",
    )
    gene_name: str | None = Field(
        default=None,
        description="The gene symbol named in the message. Empty when no gene is named.",
    )
    mutation: str | None = Field(
        default=None,
        description="Explicit point mutation such as R273H. Empty when none is named.",
    )
    residue_position: int | None = Field(
        default=None,
        gt=0,
        description="Explicit positive residue position, excluding a position only present in mutation.",
    )
    accession: str | None = Field(
        default=None,
        description="Nucleotide database accession written in the message, e.g. "
        "'NC_007596'. Empty when the message names none.",
    )
    requested_regions: list[str] = Field(
        default_factory=list,
        description="Explicit protein domains or regions requested by the caller.",
    )
    preferred_source: Literal["auto", "pdb", "alphafold"] | None = Field(
        default=None,
        description="Explicit structure-source preference. Empty when the caller has no preference.",
    )
    include_explanation: bool | None = Field(
        default=None,
        description="Explicit explanation preference. Empty when the caller did not specify one.",
    )


class Extractor:
    """Pulls the named entities out of a user's message."""

    def __init__(self) -> None:
        self._chain = _PROMPT | get_llm().with_structured_output(_ExtractorOutput)

    def extract(self, user_query: str) -> dict[str, Any]:
        """Return the entities named in `user_query`, ready to merge into context.

        Only fields the model actually filled in come back. A key that is
        absent is meaningfully different from a key set to None: agents test
        with `"species" not in context` as well as `context.get("species")`,
        so writing a blank would satisfy the first check while failing the
        second. Returning nothing at all keeps both honest.
        """

        response = cast(
            _ExtractorOutput,
            self._chain.invoke({"user_query": user_query}),
        )

        facts: dict[str, Any] = {
            key: value.strip()
            for key, value in (
                ("species", response.species),
                ("trait_name", response.trait_name),
                ("gene_name", response.gene_name),
                ("mutation", response.mutation),
                ("accession", response.accession),
            )
            # `value.strip()` below would blow up on None, and some models
            # answer with "" or "none" instead of leaving the field out.
            if value
            and value.strip()
            and value.strip().lower() not in {"none", "n/a", "unknown"}
        }

        if "mutation" in facts:
            facts["mutation"] = facts["mutation"].upper()
        if response.residue_position is not None:
            facts["residue_position"] = response.residue_position

        regions = [
            region.strip() for region in response.requested_regions if region.strip()
        ]
        if regions:
            facts["requested_regions"] = regions
        if response.preferred_source is not None:
            facts["preferred_source"] = response.preferred_source
        if response.include_explanation is not None:
            facts["include_explanation"] = response.include_explanation

        # Found by regex rather than by the model - see `_find_sequence`. This
        # is the key the Reconstruction agent repairs, and without it a request
        # to fill in a pasted sequence reached that agent with nothing to work
        # on and came straight back as a failure.
        sequence = _find_sequence(user_query)
        if sequence:
            facts["sequence"] = sequence

        if facts:
            # `sequence` can be tens of kilobases; logging it in full buries
            # every other line in the console the orchestrator streams to.
            loggable = {
                key: (f"<{len(value)} bases>" if key == "sequence" else value)
                for key, value in facts.items()
            }
            _logger.info("[Extractor] query=%r -> %s", user_query, loggable)
        else:
            _logger.info("[Extractor] query=%r -> no entities named", user_query)

        return facts
