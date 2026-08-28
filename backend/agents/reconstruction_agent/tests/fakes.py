"""Test doubles for the external services.

Hand-written rather than generated from recorded traffic, so a test states the
biology it depends on in the test itself. The lineage below is a real fragment
of the NCBI tree, which is what lets the search-scope tests assert the
*mechanism* - narrower clades are tried first - without a network call.
"""

from __future__ import annotations

from reconstruction_agent.config.settings import (
    AppSettings,
    AzureOpenAISettings,
    EmblEbiSettings,
    Environment,
    HomologySettings,
    HttpSettings,
    LlmSettings,
    NcbiBlastSettings,
    NcbiSettings,
    NvidiaSettings,
    ObservabilitySettings,
    ReconstructionSettings,
    Settings,
)
from reconstruction_agent.domain.models.taxonomy import TaxonNode

# A real fragment of NCBI Taxonomy, root-first, as `lineage` returns it.
VERTEBRATA = TaxonNode(tax_id=7742, name="Vertebrata", rank="clade")
MAMMALIA = TaxonNode(tax_id=40674, name="Mammalia", rank="class")
CARNIVORA = TaxonNode(tax_id=33554, name="Carnivora", rank="order")
URSIDAE = TaxonNode(tax_id=9632, name="Ursidae", rank="family")
URSUS_MARITIMUS = TaxonNode(tax_id=29073, name="Ursus maritimus", rank="species")

RODENTIA = TaxonNode(tax_id=9989, name="Rodentia", rank="order")
AVES = TaxonNode(tax_id=8782, name="Aves", rank="class")
#: The insect genus that NCBI returns ahead of the domain when searching the
#: bare name "Bacteria" - the homonym the selector has to survive.
BACTERIA_INSECT = TaxonNode(tax_id=629395, name="Bacteria Latreille", rank="genus")
BACTERIA_DOMAIN = TaxonNode(tax_id=2, name="Bacteria", rank="domain")
#: Resolves cleanly and contains no animal - the "positively wrong" case.
FUNGI = TaxonNode(tax_id=4751, name="Fungi", rank="kingdom")

POLAR_BEAR_LINEAGE = (VERTEBRATA, MAMMALIA, CARNIVORA, URSIDAE, URSUS_MARITIMUS)


class FakeTaxonomyClient:
    """Name resolution over a fixed table, with the same interface as the real one.

    Keys are the group terms a catalogue label yields. Values are ordered as
    NCBI orders them, most relevant first, so a homonym can be placed ahead of
    the correct answer exactly as it is live.
    """

    def __init__(self, names: dict[str, tuple[TaxonNode, ...]] | None = None) -> None:
        self.names: dict[str, tuple[TaxonNode, ...]] = (
            names
            if names is not None
            else {
                "Mammal": (MAMMALIA,),
                "Vertebrate": (VERTEBRATA,),
                "Rodent": (RODENTIA,),
                "Bird": (AVES,),
                # Deliberately homonym-first.
                "Bacteria": (BACTERIA_INSECT, BACTERIA_DOMAIN),
                "Fungi": (FUNGI,),
                # Paraphyletic; NCBI resolves nothing for it.
                "Invertebrate": (),
            }
        )
        self.lookups: list[str] = []

    async def resolve_candidates(self, name: str, *, limit: int = 3) -> tuple[TaxonNode, ...]:
        self.lookups.append(name)
        return self.names.get(name, ())[:limit]

    async def resolve_name(self, name: str) -> TaxonNode | None:
        candidates = await self.resolve_candidates(name)
        return candidates[0] if candidates else None


def isolated_settings(**overrides: object) -> Settings:
    """Settings built with the environment and .env file deliberately ignored.

    Without this a suite silently reads whatever .env the developer happens to
    have, so a test asserting that a dependency is unconfigured passes or fails
    according to which API keys are on the machine running it. Every sub-model
    is constructed with `_env_file=None` and no environment source, which is the
    only way to get a genuinely fixed configuration.
    """
    base: dict[str, object] = {
        "app": AppSettings(_env_file=None, env=Environment.TEST),
        "observability": ObservabilitySettings(_env_file=None),
        "http": HttpSettings(_env_file=None),
        "ncbi": NcbiSettings(_env_file=None),
        "embl_ebi": EmblEbiSettings(_env_file=None),
        "nvidia": NvidiaSettings(_env_file=None),
        "azure_openai": AzureOpenAISettings(_env_file=None),
        "llm": LlmSettings(_env_file=None),
        "ncbi_blast": NcbiBlastSettings(_env_file=None),
        "homology": HomologySettings(_env_file=None),
        "reconstruction": ReconstructionSettings(_env_file=None),
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]
