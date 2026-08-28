"""Wiring for the HTTP layer.

Settings come from the application instance rather than the process-wide cache.
That distinction matters: an endpoint calling `get_settings()` directly would
read whatever .env is on the machine, so an application built from explicit
settings would be configured one way and answer requests another - a failure
that only shows up on somebody else machine.

The service graph is built once and shared. Pacing lives on each HTTP client
instance, so two clients for one upstream each honour the rate limit separately
and together exceed it; a per-request graph would also discard the taxonomy and
catalogue caches that keep a run inside its deadline.
"""

from __future__ import annotations

from fastapi import Request

from reconstruction_agent.agent.runner import GapRunner
from reconstruction_agent.config.settings import Settings, get_settings
from reconstruction_agent.integrations.evo2.client import Evo2Client
from reconstruction_agent.integrations.llm import build_llm_client
from reconstruction_agent.integrations.mafft.client import MafftClient
from reconstruction_agent.integrations.ncbi.client import NcbiClient
from reconstruction_agent.integrations.ncbi.taxonomy import TaxonomyClient
from reconstruction_agent.integrations.ncbi_blast.client import NcbiBlastClient
from reconstruction_agent.orchestration.deadline import PhaseBudget
from reconstruction_agent.services.candidate.candidate_builder import CandidateBuilder
from reconstruction_agent.services.homology.homology_service import HomologyService
from reconstruction_agent.services.plausibility.evo2_service import Evo2Service
from reconstruction_agent.services.reconstruction.reconstruction_service import (
    ReconstructionService,
)
from reconstruction_agent.services.scoring.confidence_engine import (
    ConfidenceEngine,
    ConfidenceThresholds,
)
from reconstruction_agent.services.sequence.sequence_service import SequenceService
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService
from reconstruction_agent.tools import build_registry


def get_settings_dependency(request: Request) -> Settings:
    """The settings the running application was built with."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def build_service(settings: Settings) -> ReconstructionService:
    """Construct the reconstruction service and everything beneath it."""
    ncbi = NcbiClient(settings.ncbi, timeout=settings.http.timeout_seconds)
    taxonomy_client = TaxonomyClient(ncbi, cache_ttl=settings.ncbi.cache_ttl_seconds)
    # Alignment is the one thing still run at EMBL-EBI: NCBI publishes no
    # alignment service and no MAFFT binary is installed in this repository.
    mafft = MafftClient(settings.embl_ebi, timeout=settings.http.timeout_seconds)

    taxonomy = TaxonomyService(taxonomy_client)
    engine = ConfidenceEngine(
        thresholds=ConfidenceThresholds(minimum=settings.reconstruction.min_confidence)
    )
    sequences = SequenceService(ncbi)
    candidates = CandidateBuilder(engine=engine, taxonomy=taxonomy)
    homology = HomologyService(
        blast=NcbiBlastClient(
            settings.ncbi_blast,
            # The BLAST endpoint answers slowly under load and a submission
            # is a POST carrying the whole query, so it gets a longer
            # per-request timeout than the sequence and taxonomy calls.
            timeout=settings.http.timeout_seconds * 2,
        ),
        taxonomy=taxonomy_client,
        settings=settings.ncbi_blast,
        homology=settings.homology,
    )
    # Evo 2 is constructed unconditionally but connects lazily: arbitration is
    # conditional, so most runs never reach it, and an unconfigured key makes
    # the tool decline with a reason rather than disappear from the surface.
    evo2 = Evo2Service(Evo2Client(settings.nvidia))
    phases = PhaseBudget(
        homology_seconds=settings.reconstruction.homology_budget_seconds,
        alignment_seconds=settings.reconstruction.alignment_budget_seconds,
        arbitration_seconds=settings.reconstruction.arbitration_budget_seconds,
    )

    return ReconstructionService(
        settings=settings,
        sequences=sequences,
        taxonomy=taxonomy,
        homology=homology,
        mafft=mafft,
        candidates=candidates,
        engine=engine,
        # The agent loop. Every tool it can select is built over the same
        # services above, so the graph and the deterministic pipeline reach
        # identical biology by different routes.
        runner=GapRunner(
            registry=build_registry(
                sequences=sequences,
                taxonomy=taxonomy,
                homology=homology,
                mafft=mafft,
                candidates=candidates,
                engine=engine,
                evo2=evo2,
            ),
            llm=build_llm_client(settings),
            engine=engine,
            phases=phases,
        ),
    )


def get_reconstruction_service(request: Request) -> ReconstructionService:
    """The shared reconstruction service, built on first use."""
    service = getattr(request.app.state, "reconstruction_service", None)
    if isinstance(service, ReconstructionService):
        return service

    service = build_service(get_settings_dependency(request))
    request.app.state.reconstruction_service = service
    return service
