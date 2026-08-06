"""Deterministic analysis of the instruction. No LLM.

Text does two jobs in this agent, and only two:

1. It says what the user wants - identification, visually similar species, or a
   scientific follow-up that another agent owns.
2. It supplies hints (a suspected taxon, a location, a habitat) that can agree
   with, or contradict, what the image retrieved.

It never supplies a species. A name in the instruction is compared against the
candidates retrieval already produced; if it matches none of them, that is a
conflict to be reported, not a new candidate to be added. This is the single
rule that keeps the image as the primary evidence.

Everything here is rule-based. Clear intents cost nothing and are reproducible,
which is exactly what the specification asks for before spending an LLM call.
GPT-5 mini is the approved reasoning model for this agent, but wiring it in is a
separate authorised step - and the rules below stay as its fallback either way.
"""
from __future__ import annotations

import logging
import re

from .adapters.reasoning_llm import (
    ReasoningBudget,
    ReasoningLLM,
    ReasoningRequest,
)
from .domain.models import SpeciesCandidate, TextAlignment, TextEvidence

_logger = logging.getLogger(__name__)

# Intent keywords. Ordered by specificity: a follow-up question wins over a
# similarity question, which wins over plain identification.
_FOLLOW_UP_CAPABILITIES: dict[str, tuple[str, ...]] = {
    # capability hint -> trigger words
    "Evolution": ("evolution", "evolutionary", "phylogen", "ancestor", "ancestry",
                  "lineage", "diverge", "related to", "tree of life"),
    "Genome": ("genome", "genomic", "dna", "chromosome", "assembly", "sequence"),
    "Biodiversity": ("habitat", "distribution", "range", "conservation", "endangered",
                     "iucn", "population", "migration"),
    "Trait": ("trait", "adaptation", "morpholog", "behaviour", "behavior"),
    "Literature": ("paper", "publication", "literature", "study", "studies", "pubmed"),
    "Protein": ("protein", "structure", "alphafold", "uniprot"),
}

_SIMILARITY_WORDS = ("similar", "resembl", "looks like", "look like", "close relative",
                     "lookalike", "look-alike", "comparable")

# "Panthera leo" - genus capitalised, species lowercase.
#
# The pattern alone is not enough: an English sentence starting "Could this be"
# or "Please identify" matches it perfectly and would invent a taxon hint out of
# ordinary prose. So a capitalised word that is simply how a sentence begins is
# excluded. A false hint is worse than no hint - it ends up in the output.
_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b")

_GENUS_STOPWORDS = frozenset({
    "could", "should", "would", "what", "which", "where", "when", "this", "that",
    "these", "those", "there", "here", "please", "identify", "show", "tell",
    "find", "does", "did", "are", "can", "may", "might", "must", "will", "the",
    "and", "but", "also", "after", "before", "given", "with", "from", "your",
    "does", "how", "why", "who", "whose", "look", "looks", "seems", "maybe",
    "perhaps", "assume", "confirm", "check", "compare", "describe", "explain",
    "based", "using", "about", "there", "some", "many", "most", "each", "every",
})

_LOCATION_RE = re.compile(
    r"\b(?:in|from|near|around)\s+((?:the\s+)?[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*)"
)

_HABITAT_WORDS = ("savanna", "savannah", "rainforest", "forest", "desert", "tundra",
                  "arctic", "grassland", "wetland", "mountain", "coastal", "marine",
                  "jungle", "steppe")

# Phrases that make the intent unambiguous on their own. Their absence is what
# defines an "ambiguous" request - the only case where an enabled reasoning model
# is consulted at all.
_RECOGNITION_TRIGGERS = ("identify", "identification", "what animal", "what species",
                         "which animal", "which species", "what is this", "what's this",
                         "name this", "name the animal", "recognise", "recognize",
                         "what kind of animal", "what type of animal")

# --- deterministic language detection -------------------------------------
#
# Marker words are chosen to be EXCLUSIVE to one language, so a word that exists
# in both (e.g. "animal", "photo") contributes nothing and cannot tip the result.
# When the evidence is thin or split, the answer is None: reporting the wrong
# language is worse than reporting none, and nothing in the workflow reads it.
_FRENCH_MARKERS = frozenset({
    "est", "ce", "cet", "cette", "quel", "quelle", "quels", "quelles", "espèce",
    "espece", "identifie", "identifier", "dans", "avec", "pour", "les", "des",
    "une", "un", "le", "la", "du", "de", "que", "qui", "quoi", "vous", "plaît",
    "plait", "montre", "donne", "sur", "oiseau", "prise", "cliché", "cliche",
    "s'il", "peux", "peut", "quelles", "sont", "était", "voici",
})
_ENGLISH_MARKERS = frozenset({
    "what", "which", "this", "that", "is", "the", "identify", "species",
    "please", "tell", "show", "find", "does", "are", "was", "were", "would",
    "could", "picture", "taken", "these", "those", "here", "there", "about",
    "kind", "type", "your", "my",
})
# Letters that effectively only appear in French among the three supported
# languages - a cheap, high-confidence signal.
_FRENCH_LETTERS = frozenset("àâäçéèêëîïôöùûüÿœæ")

_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                  (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

_WORD_RE = re.compile(r"[\w'’À-ÿ]+", re.UNICODE)


def detect_language(instruction: str) -> str | None:
    """Best-effort `en` / `fr` / `ar`, or None. Deterministic, no dependency."""

    if not instruction or not instruction.strip():
        return None

    # Script beats vocabulary: Arabic characters are unambiguous.
    arabic = sum(
        1 for char in instruction
        if any(low <= ord(char) <= high for low, high in _ARABIC_RANGES)
    )
    if arabic >= 2:
        return "ar"

    words = {match.group(0).lower() for match in _WORD_RE.finditer(instruction)}
    if len(words) < 2:
        return None

    french = len(words & _FRENCH_MARKERS)
    english = len(words & _ENGLISH_MARKERS)
    if any(char in _FRENCH_LETTERS for char in instruction.lower()):
        french += 2

    # Needs both a floor and a clear winner. A tie, or a single weak marker, is
    # not enough to name a language.
    if french >= 2 and french > english:
        return "fr"
    if english >= 2 and english > french:
        return "en"
    return None


class RuleBasedTextAnalyzer:
    """Turns an instruction into `TextEvidence` using fixed rules only."""

    mode = "rules"

    def __init__(self, known_names: dict[str, str] | None = None) -> None:
        # Maps a lowercase scientific or common name to a species_id. Supplied by
        # the taxonomy provider, so the analyser knows which names are *known*
        # without being able to introduce them as candidates.
        self._known_names = known_names or {}

    def classify_intent(self, lowered: str) -> tuple[str, str | None, bool]:
        """Return (intent, requested_capability, is_clear).

        `is_clear` is the whole point: it is True when an explicit trigger
        matched, and False only when the intent fell through to the
        `recognition` default. That distinction is what lets clear requests cost
        exactly zero LLM calls.
        """
        for capability, triggers in _FOLLOW_UP_CAPABILITIES.items():
            if any(trigger in lowered for trigger in triggers):
                return "scientific_follow_up", capability, True

        if any(word in lowered for word in _SIMILARITY_WORDS):
            return "similarity", None, True

        if any(trigger in lowered for trigger in _RECOGNITION_TRIGGERS):
            return "recognition", None, True

        # Nothing matched. Default to recognition - the safe assumption for an
        # agent that was handed an image - but mark it ambiguous.
        return "recognition", None, False

    def analyze(
        self,
        instruction: str,
        *,
        llm: ReasoningLLM | None = None,
        budget: ReasoningBudget | None = None,
        candidates: tuple[str, ...] = (),
    ) -> TextEvidence:
        """Rules first, always. The model is a refinement, never a prerequisite.

        `llm` and `budget` are optional: called without them - as every existing
        caller does - this is exactly the deterministic analyser it always was.
        """
        lowered = instruction.lower()

        intent, requested_capability, is_clear = self.classify_intent(lowered)
        evidence = TextEvidence(
            intent=intent,
            language=detect_language(instruction),
            taxon_hint=self._taxon_hint(instruction, lowered),
            location_hint=self._location_hint(instruction),
            habitat_hint=next((word for word in _HABITAT_WORDS if word in lowered), None),
            requested_capability=requested_capability,
        )

        # Four independent reasons to spend nothing: no adapter, a disabled one,
        # a clear intent, or an exhausted budget. Only an ambiguous request with
        # an enabled adapter and budget left gets a call.
        if llm is None or not getattr(llm, "enabled", False) or is_clear:
            return evidence
        if budget is None or not budget.consume():
            return evidence

        try:
            refined = llm.analyze(
                ReasoningRequest(
                    instruction=instruction,
                    rule_intent=intent,
                    candidate_names=tuple(candidates),
                    candidate_species_ids=(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            # The bundled adapter contains its own failures, but this must hold
            # for ANY adapter, including one someone else supplies later. A
            # reasoning model is an optional refinement: nothing it does may
            # turn a working request into a failed one.
            _logger.info(
                "[Recognition] reasoning adapter raised (%s); using deterministic rules.",
                type(exc).__name__,
            )
            return evidence

        if refined is None:
            # Timeout, provider error, or off-contract output. Rules stand.
            return evidence

        try:
            return self._merge(evidence, refined, budget)
        except Exception as exc:  # noqa: BLE001
            # A result shaped unlike ReasoningResult must not break the request.
            _logger.info(
                "[Recognition] reasoning result unusable (%s); using deterministic rules.",
                type(exc).__name__,
            )
            return evidence

    @staticmethod
    def _merge(
        evidence: TextEvidence, refined, budget: ReasoningBudget
    ) -> TextEvidence:
        """Fold an accepted model result into the rule result.

        The model may only refine the intent and FILL hints the rules left
        empty - it can never overwrite something the rules found in the text.
        A hint it supplies is still just a string; it becomes meaningful only
        after `resolve_hint` matches it against known species, so this cannot
        introduce a candidate.
        """
        updates: dict[str, object] = {}

        if refined.intent is not None and refined.intent != evidence.intent:
            updates["intent"] = refined.intent
        for name in ("taxon_hint", "location_hint", "habitat_hint", "language"):
            supplied = getattr(refined, name, None)
            if supplied is not None and getattr(evidence, name) is None:
                updates[name] = supplied

        if not updates:
            return evidence

        budget.accepted = True
        return evidence.model_copy(update=updates)

    def _taxon_hint(self, instruction: str, lowered: str) -> str | None:
        """A species name the user appears to have named."""

        # A known name is the strongest signal. Longest first, so "african bush
        # elephant" wins over "elephant" if both were known.
        for name in sorted(self._known_names, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return name

        # Otherwise a scientific binomial, if the instruction contains one that
        # is not just an English sentence opening.
        for match in _BINOMIAL_RE.finditer(instruction):
            genus, species = match.group(1), match.group(2)
            if genus.lower() in _GENUS_STOPWORDS:
                continue
            return f"{genus} {species}"
        return None

    @staticmethod
    def _location_hint(instruction: str) -> str | None:
        match = _LOCATION_RE.search(instruction)
        return match.group(1).strip() if match else None

    def resolve_hint(self, taxon_hint: str | None) -> str | None:
        """The species_id a hint refers to, if we know the name at all."""
        if not taxon_hint:
            return None
        return self._known_names.get(taxon_hint.strip().lower())


def align_text_with_candidates(
    evidence: TextEvidence,
    candidates: list[SpeciesCandidate],
    resolved_hint_species_id: str | None,
) -> TextAlignment:
    """Does the instruction agree with what the image retrieved?

    - `agree`    the named species is the top candidate;
    - `conflict` the named species is known, but is not the top candidate;
    - `neutral`  no usable name, or a name we cannot resolve.

    A conflict never removes or adds a candidate. It only prevents the workflow
    from claiming an identification it cannot support.
    """
    if not candidates:
        return "neutral"
    if evidence.taxon_hint is None:
        return "neutral"

    # An unresolvable name is not evidence of disagreement - we simply do not
    # know what the user meant, and guessing would manufacture a conflict.
    if resolved_hint_species_id is None:
        return "neutral"

    if resolved_hint_species_id == candidates[0].species_id:
        return "agree"

    # Known species, but the image put something else first - including the case
    # where it is a lower-ranked candidate. Either way the text and the image
    # disagree about what this is.
    return "conflict"
