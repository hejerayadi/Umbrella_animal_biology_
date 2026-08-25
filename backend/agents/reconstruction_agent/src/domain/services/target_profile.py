"""Which ENA division a target belongs to, and what kind of molecule it is.

The agent used to send every search to `em_std_vrt` and hope. That cannot work,
and the reason is not tuning: **EMBL's taxonomic divisions are mutually
exclusive.** VRT is *"Other Vertebrates"* - it holds the vertebrates that are not
human, not mouse, not rodent and *not mammal*. A polar bear record is filed under
MAM and is simply not present in VRT, so no e-value threshold, no retry and no
ranking change could ever have surfaced it.

Measured against the live service on 2026-08-25, on a 45-base hole in the polar
bear mitogenome `NC_003428.1`::

    em_std_vrt   50 hits,  7 carrying the gap - all fish, ~67% identity
    em_std_mam   50 hits, 50 carrying the gap - Ursus maritimus at 95.7%,
                 recovering all 45 bases exactly

Same query, same parameters, same minute. The division *is* the answer.

This module is the deterministic half of choosing one. It is a **prior**, not a
policy: it seeds the planner's shortlist and stands in when the LLM or the
network is unavailable, while the actual choice is settled by measuring
`blast_hits_carrying_gap` per candidate. Keeping it pure is what lets the offline
smoke test and the unit suite exercise the decision with no network at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Division(str, Enum):
    """An EMBL/ENA taxonomic division. Values are the codes EBI accepts."""

    HUM = "hum"
    MUS = "mus"
    ROD = "rod"
    MAM = "mam"
    VRT = "vrt"
    INV = "inv"
    PLN = "pln"
    FUN = "fun"
    PRO = "pro"
    VRL = "vrl"


class Molecule(str, Enum):
    """What kind of sequence the target is, which picks the *subdivision*.

    `em_std_*` holds finished records - a published mitogenome lives there.
    `em_*` is the whole division and also carries the WGS, CON and HTG entries
    where draft genomic scaffolds actually live. A gap in a Scaffold-level
    assembly has to look in the latter; a gap in a mitogenome should look in the
    former first, because it is smaller, cleaner and faster.
    """

    ORGANELLAR = "organellar"
    NUCLEAR = "nuclear"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """What we believe about the target, and how we came to believe it.

    `division` is None when nothing placed the organism. That is not a failure to
    paper over with a guess: an arbitrary division is worse than an empty prior,
    because the planner can probe several candidates in parallel and let the
    measurement decide, whereas a confident wrong answer sends every search to
    the one place the homologues are not.
    """

    division: Division | None
    molecule: Molecule
    #: How the division was reached: "lineage" | "description" | "none".
    #: Carried into diagnostics so a bad choice stays attributable after the fact.
    source: str = "none"


#: Lineage clade -> division, tried in this order. The order is load-bearing
#: because the divisions do not nest: a mammal is in MAM *instead of* VRT, not as
#: well as, so a broader test placed earlier would swallow everything under it.
_CLADE_RULES: tuple[tuple[str, Division], ...] = (
    ("Rodentia", Division.ROD),
    ("Mammalia", Division.MAM),
    ("Vertebrata", Division.VRT),
    ("Craniata", Division.VRT),
    ("Chordata", Division.VRT),
    ("Metazoa", Division.INV),
    ("Viridiplantae", Division.PLN),
    ("Fungi", Division.FUN),
    ("Bacteria", Division.PRO),
    ("Archaea", Division.PRO),
    ("Viruses", Division.VRL),
)

#: Organellar giveaways in a record's description.
_ORGANELLAR = re.compile(
    r"mitochondri|chloroplast|plastid|apicoplast|kinetoplast|\bmtDNA\b|\bcpDNA\b",
    re.IGNORECASE,
)

#: Draft-assembly giveaways. `unplaced` and `scaffold` both appear in the real
#: defline of `NW_007907101`: "Ursus maritimus isolate Baiyulong unplaced
#: genomic scaffold".
_NUCLEAR = re.compile(
    r"scaffold|contig|whole[ _-]genome[ _-]shotgun|\bWGS\b|unplaced|unlocalis|draft",
    re.IGNORECASE,
)

#: Assembly levels that describe a nuclear assembly. "Complete Genome" is
#: excluded on purpose - it is what NCBI reports for a finished mitogenome too,
#: so reading it as nuclear would misfile exactly the case that already works.
_NUCLEAR_ASSEMBLY_LEVELS = frozenset({"scaffold", "contig", "chromosome"})

#: Above this, a sequence is nuclear whatever it calls itself. A mitogenome is
#: ~16.5 kb and a chloroplast ~150 kb; nothing organellar reaches a megabase.
_NUCLEAR_LENGTH = 1_000_000


def classify_division(lineage: tuple[str, ...]) -> Division | None:
    """The division an organism's records are filed under, from its NCBI lineage.

    `lineage` is what `TaxonomyService.lineage` returns: the clades outermost
    first, with the organism's own scientific name appended as the leaf.

    The two leaf tests come first and are deliberately exact. HUM is *Homo
    sapiens* alone - Neanderthal and Denisovan records are filed MAM, and since
    the divisions do not nest there is no sideways recovery from guessing HUM for
    them. MUS is *Mus musculus* alone for the same reason: other *Mus* is ROD.
    """
    if not lineage:
        return None

    leaf = lineage[-1]
    if leaf.startswith("Homo sapiens"):
        return Division.HUM
    if leaf.startswith("Mus musculus"):
        return Division.MUS

    clades = set(lineage)
    for clade, division in _CLADE_RULES:
        if clade in clades:
            return division

    return None


def classify_molecule(
    *,
    description: str | None = None,
    assembly_level: str | None = None,
    length: int | None = None,
) -> Molecule:
    """Organellar or nuclear, from whichever signals the caller actually has.

    Three independent sources, because no single one is always present: the
    record's own description, the assembly level the Genome Agent publishes in
    `genome_metadata`, and the raw sequence length. Organellar is tested first -
    a mitogenome's description says so outright, and that beats any inference.

    UNKNOWN is a real answer, not a failure. It means "no signal either way", and
    the caller treats it as finished-records-first, which is the cheaper search.
    """
    text = description or ""
    if _ORGANELLAR.search(text):
        return Molecule.ORGANELLAR

    level = (assembly_level or "").strip().casefold()
    if level in _NUCLEAR_ASSEMBLY_LEVELS:
        return Molecule.NUCLEAR

    if _NUCLEAR.search(text):
        return Molecule.NUCLEAR

    if length is not None and length >= _NUCLEAR_LENGTH:
        return Molecule.NUCLEAR

    return Molecule.UNKNOWN


def candidate_databases(profile: TargetProfile) -> tuple[str, ...]:
    """The databases worth probing for this target, best guess first.

    Composed from the division rather than picked out of a hardcoded table, and
    validated against EBI's live catalogue before anything is submitted - see
    `tools.blast.catalogue`. An empty tuple means the prior has nothing to offer
    and the planner should decide unaided.

    Deliberately excludes `em_std` and `em_all`. Both were measured against the
    real service and both blow the 600 s orchestrator slice - `em_all` took 749 s
    and `em_std` had not finished at 813 s - so a "search everything" fallback is
    not a safety net, it is a guaranteed timeout.
    """
    if profile.division is None:
        return ()

    division = profile.division.value
    if profile.molecule is Molecule.NUCLEAR:
        # Draft scaffolds live in WGS/CON/HTG, which only the whole division has.
        return (f"em_{division}", f"em_std_{division}")
    return (f"em_std_{division}", f"em_{division}")


#: Status prefixes INSDC deflines carry ahead of the organism name. Left in
#: place they would be read as the genus.
_ORGANISM_PREFIX = re.compile(
    r"^(?:UNVERIFIED(?:_ORG)?|TPA(?:_inf|_exp)?|MAG|PREDICTED)\s*:\s*",
    re.IGNORECASE,
)
_BINOMIAL = re.compile(r"^([A-Z][a-z]+ [a-z]+)")
#: Placeholders that parse as a binomial but name no organism.
_NOT_AN_ORGANISM = frozenset({"uncultured", "unidentified", "synthetic", "unclassified"})


def organism_from_description(description: str | None) -> str | None:
    """The binomial at the front of an INSDC defline, or None.

    BLAST's own organism field is unusable: `hit_os` came back as the literal
    string `"NA"` on all 200 hits across four real searches, so every reference
    reached the ranker with no organism at all - which is why the ranker's
    relatedness weight had been silently multiplying zero. The name is in the
    description the whole time: "Ursus maritimus isolate PB18-N26025
    mitochondrion, complete genome."
    """
    if not description:
        return None

    text = _ORGANISM_PREFIX.sub("", description.strip())
    match = _BINOMIAL.match(text)
    if not match:
        return None

    name = match.group(1)
    if name.split(" ")[0].casefold() in _NOT_AN_ORGANISM:
        return None
    return name
