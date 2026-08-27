"""The workflow's nodes.

Each node is built by a factory that captures the collaborators it needs, then
returns a plain function `(state) -> dict of updates`. That is the LangGraph
node shape, and it is also what makes every node testable on its own: build it
with a stub, hand it a state, inspect what it returned.

Nodes never raise. A `RecognitionError` is caught and written into the state as
`error_code`, and the router sends the graph straight to `finalize` - so there
is exactly one place an `AgentResult` is built, whatever happened.

No node calls another agent. When this agent needs help it says so in the
result, and the Global Orchestrator decides who provides it.

The order the evidence is built in is the point of the whole file:

    validate -> plan -> CLASSIFY -> decide -> validate taxonomy -> explain

The species candidates exist only after `classify_with_mock_bioclip2`, and the
confidence gate closes before the taxonomy sources are consulted at all. Neither
the text, nor the reasoning model, nor GBIF, nor NCBI has any point at which it
could add a taxon to that list.

--- Sprint 4 Phase 5 -------------------------------------------------------

`plan`, `classify`, `taxonomy` and `explain` each accept an optional
`RecognitionTracer` (see `..observability`) and, at the end of their work,
record one sanitized event describing what actually happened: which provider
or LLM role ran, whether it fell back to the deterministic path, and (for
classify) the fixed error code on a controlled failure. Nothing else about a
node's behaviour changes. Tracing is disabled by default - every node factory
defaults to a shared, disabled `_NULL_TRACER`, so a call site or test that
predates Phase 5 needs no change and reaches no client, no network and no
extra work at all when tracing is off. LangGraph's own tracing already covers
the node-to-node skeleton when LangSmith is configured; the events recorded
here are the provider-level detail LangGraph's tracing cannot see on its own.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..adapters.bioclip import BioCLIP2Classifier
from ..adapters.reasoning_llm import (
    ExplainRequest,
    PlanRequest,
    ReasoningBudget,
    deterministic_plan,
    explanation_is_grounded,
    sanitize_plan,
)
from ..adapters.taxonomy import MockTaxonomyProvider
from ..config import RECOGNITION_MODE_MOCK_CLASSIFICATION, LangSmithConfig, RecognitionConfig
from ..domain import confidence, ranking
from ..domain.errors import RecognitionError
from ..domain.models import RecognitionDecision
from ..observability import RecognitionTracer
from ..text_analysis import RuleBasedTextAnalyzer, align_text_with_candidates
from ..validation import validate_paired_request
from .state import HELPER_OUTPUT_KEYS, RecognitionState

# Logs the workflow's decisions and nothing else. No image, no pixel, no
# context value ever reaches a log record - see the leak tests.
_logger = logging.getLogger(__name__)

Node = Callable[[RecognitionState], dict]

# What the response says when the instruction asked for something this agent
# does not provide. There is one such thing, and this is where it is named.
UNSUPPORTED_SIMILARITY_CAPABILITY = "visual_similarity_search"

# The default when a node factory is called without a tracer - e.g. by an
# existing test that predates Phase 5. Disabled, so it builds no client and
# makes no call; every `.record()` on it is a no-op. One shared instance, not
# one per node: it holds no per-request state, so sharing it is safe.
_NULL_TRACER = RecognitionTracer(LangSmithConfig())


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _failed(exc: RecognitionError) -> dict:
    return {"error_code": exc.code.value, "error_message": exc.message}


# --- what the taxonomy sources actually were --------------------------------

def taxonomy_source_modes(state: RecognitionState) -> tuple[str, str, bool]:
    """`(gbif_mode, ncbi_mode, executed)`, from what actually ran.

    Runtime evidence first: every provider - mock or real - records the mode of
    each source in the per-species `taxonomy_report` it returns, so that report
    is the truth about which services answered this request. Configuration is
    never consulted, which is what makes an injected or swapped provider report
    honestly.

    When the classifier named nothing there is no candidate to validate, so the
    report is empty and no lookup happened. That is reported as `executed=False`
    alongside the mode the wired provider declares about *itself* - saying which
    sources would have been consulted is honest; claiming they answered is not.
    A provider that declares nothing yields `"unknown"` rather than a guess.
    """
    report = state.taxonomy_report or {}

    def modes_for(source: str) -> set[str]:
        found = set()
        for entry in report.values():
            if isinstance(entry, dict) and isinstance(entry.get(source), dict):
                mode = entry[source].get("mode")
                if isinstance(mode, str) and mode:
                    found.add(mode)
        return found

    gbif_modes, ncbi_modes = modes_for("gbif"), modes_for("ncbi")
    executed = bool(gbif_modes or ncbi_modes)

    if not executed:
        declared = (state.config_snapshot or {}).get("taxonomy_provider_mode")
        declared = declared if isinstance(declared, str) and declared else "unknown"
        return declared, declared, False

    def one(found: set[str]) -> str:
        if len(found) == 1:
            return next(iter(found))
        # Two sources of the same name reporting different modes in one request
        # is not something to average away - it is stated as what it is.
        return "mixed" if found else "unknown"

    return one(gbif_modes), one(ncbi_modes), True


def _source_label(source: str, mode: str, *, sentence_start: bool = False) -> str:
    """How to name one taxonomy source in prose, given the mode that ran.

    `sentence_start` capitalises only the article. `str.capitalize()` would
    lowercase the rest of the string and turn "NCBI" into "ncbi".
    """
    article = "The" if sentence_start else "the"
    if mode == "mock":
        return f"{article} mocked {source} source"
    if mode == "real":
        return f"{article} live {source} lookup"
    return f"{article} {source} source"


# --- 1. validate_image_and_text --------------------------------------------

def make_validate_node(config: RecognitionConfig) -> Node:
    """Paired validation and safe decoding. Both parts are required."""

    def _node(state: RecognitionState) -> dict:
        try:
            normalized = validate_paired_request(
                state.instruction, state.context, config.validation
            )
        except RecognitionError as exc:
            return _failed(exc)

        _logger.info(
            "[Recognition] validated: %s %dx%d %d bytes, instruction present",
            normalized.media_type, normalized.width, normalized.height,
            normalized.byte_size,
        )
        return {"normalized": normalized}

    return _node


# --- 2. plan_or_analyze_text (LLM call 1 of 2) -----------------------------

def make_plan_node(
    analyzer: RuleBasedTextAnalyzer, llm: Any, config: RecognitionConfig,
    tracer: RecognitionTracer = _NULL_TRACER,
) -> Node:
    """Deterministic evidence first, then let the model plan on top of it.

    The rules always run: they produce the evidence the planner is given, and
    they are the plan used whenever the model is absent, fails, or answers
    off-contract. The model can choose which optional steps run and refine the
    hints - it cannot skip a mandatory step or invent a route.
    """

    def _node(state: RecognitionState) -> dict:
        started = time.perf_counter()
        assert state.normalized is not None
        evidence = analyzer.analyze(state.normalized.instruction)
        default_top_k = config.top_k_species

        budget = ReasoningBudget(max_calls=config.reasoning_llm_max_calls_per_request)
        plan = None
        rejected = False
        calls = 0

        if getattr(llm, "enabled", False) and hasattr(llm, "plan") and budget.consume():
            calls = 1
            try:
                raw = llm.plan(
                    PlanRequest(
                        instruction=state.normalized.instruction,
                        has_image=True,
                        image_media_type=state.normalized.media_type,
                        rule_intent=evidence.intent,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a planner may not be well-behaved
                _logger.info(
                    "[Recognition] planner raised (%s); deterministic plan used.",
                    type(exc).__name__,
                )
                raw = None

            if raw is not None:
                plan = sanitize_plan(
                    raw, default_top_k=default_top_k, rule_intent=evidence.intent
                )
                # A plan naming a forbidden step, or missing a mandatory one, is
                # rejected whole - never patched up.
                rejected = plan is None
                if rejected:
                    _logger.info("[Recognition] plan rejected by validation; using rules.")

        if plan is None:
            plan = deterministic_plan(
                intent=evidence.intent, top_k=default_top_k, evidence=evidence
            )

        # The plan may refine hints, but only where the rules found nothing.
        merged = evidence.model_copy(update={
            name: getattr(plan, name)
            for name in ("taxon_hint", "location_hint", "habitat_hint", "language",
                         "requested_capability")
            if getattr(plan, name) is not None and getattr(evidence, name) is None
        } | ({"intent": plan.intent} if plan.source == "llm" else {}))

        updates: dict[str, Any] = {
            "text_evidence": merged,
            "resolved_hint_species_id": analyzer.resolve_hint(merged.taxon_hint),
            "plan": plan,
            "plan_source": plan.source,
            "plan_rejected": rejected,
            "llm_plan_calls": calls,
            "reasoning_llm_calls": calls,
            "reasoning_llm_used": plan.source == "llm",
        }
        if merged.unsupported_capability:
            # Said out loud rather than silently ignored. The request is still
            # answered by species classification; the part this agent does not
            # own is declined, not quietly reinterpreted.
            updates["warnings"] = [
                *state.warnings,
                "This request asked for visually similar animals or images. That is not a "
                "capability of the Recognition Agent, which classifies one image into "
                "taxonomic labels; only the species identification was performed.",
            ]

        _logger.info(
            "[Recognition] plan source=%s intent=%s steps=%d llm_calls=%d",
            plan.source, merged.intent, len(plan.steps), calls,
        )
        tracer.record({
            "node": "plan",
            "operation": "plan_or_analyze_text",
            "duration_ms": _ms_since(started),
            "llm_role": "planner",
            "llm_call_count": calls,
            "fallback": plan.source != "llm",
        })
        return updates

    return _node


# --- 3. classify_with_mock_bioclip2 ----------------------------------------

def make_classify_node(
    classifier: BioCLIP2Classifier, config: RecognitionConfig,
    tracer: RecognitionTracer = _NULL_TRACER,
) -> Node:
    """The ONLY source of species candidates in this agent.

    One image goes in, an already-ranked list of taxonomic labels comes back.
    Nothing else in the workflow may add to that list, and there is no second
    path - no vector search, no reference lookup - through which a taxon could
    arrive.
    """

    def _node(state: RecognitionState) -> dict:
        started = time.perf_counter()
        assert state.normalized is not None
        # A plan may narrow K, never widen it past the configured maximum.
        planned = getattr(state.plan, "top_k", None) or config.top_k_species
        top_k = min(planned, config.top_k_species)
        provider_mode = getattr(classifier, "recognition_mode", None)

        try:
            predictions = classifier.classify(state.normalized, top_k)
            candidates = ranking.build_candidates(list(predictions), top_k=top_k)
        except RecognitionError as exc:
            # Only the fixed error code - never the exception body, which
            # could quote a malformed fixture or an upstream response.
            tracer.record({
                "node": "classify",
                "operation": "classify_with_mock_bioclip2",
                "duration_ms": _ms_since(started),
                "bioclip_provider_mode": provider_mode,
                "error_code": exc.code.value,
            })
            return _failed(exc)

        _logger.info(
            "[Recognition] classified provider=%s mode=%s labels=%d top_k=%d",
            getattr(classifier, "provider_name", "unknown"),
            getattr(classifier, "recognition_mode", "unknown"),
            len(candidates), top_k,
        )
        tracer.record({
            "node": "classify",
            "operation": "classify_with_mock_bioclip2",
            "duration_ms": _ms_since(started),
            "bioclip_provider_mode": provider_mode,
            "candidate_count": len(candidates),
        })
        return {
            "predictions": list(predictions),
            "candidates": candidates,
            "margin": ranking.top_margin(candidates),
            "visual_evidence_sufficient": confidence.visual_evidence_is_sufficient(
                candidates, config.thresholds
            ),
            "classification_provider": getattr(classifier, "provider_name", None),
            "classification_mode": getattr(classifier, "recognition_mode", None),
            "classifier_version": getattr(classifier, "version", None),
            # Whatever the provider chooses to disclose about itself.
            "classification_provenance": (
                classifier.provenance() if hasattr(classifier, "provenance") else None
            ),
            "requested_top_k": top_k,
        }

    return _node


# --- 4. evaluate_confidence -------------------------------------------------

def make_confidence_node(config: RecognitionConfig) -> Node:
    """Fuse text with the classification, then apply the confidence gate.

    Entirely deterministic, and closed before taxonomy runs. The model
    contributed the plan and will phrase the explanation; it has no part in what
    is decided here, and neither has GBIF or NCBI.
    """

    def _node(state: RecognitionState) -> dict:
        assert state.text_evidence is not None
        alignment = align_text_with_candidates(
            state.text_evidence, state.candidates, state.resolved_hint_species_id
        )
        outcome = confidence.decide(
            state.candidates, state.margin, alignment, config.thresholds
        )
        primary = (
            state.candidates[0]
            if (outcome != "not_identified" and state.candidates)
            else None
        )
        decision = RecognitionDecision(
            decision=outcome,
            primary_species=primary,
            candidates=state.candidates,
            text_alignment=alignment,
            explanation="",  # written two nodes later
            clarification_question=confidence.clarification_for(outcome, state.candidates),
            request_better_image=outcome == "not_identified",
        )
        _logger.info(
            "[Recognition] decision=%s alignment=%s candidates=%d margin=%s",
            outcome, alignment, len(state.candidates),
            None if state.margin is None else round(state.margin, 4),
        )
        return {"decision": decision}

    return _node


# --- 5. validate_taxonomy_with_mock_gbif_and_ncbi --------------------------

def make_taxonomy_node(
    provider: MockTaxonomyProvider, tracer: RecognitionTracer = _NULL_TRACER,
) -> Node:
    """Validate and normalise through both mocked sources.

    Deliberately downstream of the confidence gate: neither source can pick a
    species, change a score, change the order or change the decision. They may
    only annotate candidates the classifier already produced. A missing
    identifier stays null - never invented.
    """

    def _node(state: RecognitionState) -> dict:
        started = time.perf_counter()
        decision = state.decision
        enriched = []
        report: dict[str, Any] = {}
        degraded = False

        for candidate in state.candidates:
            if hasattr(provider, "validate_candidate"):
                updated, per_species = provider.validate_candidate(candidate)
                report[candidate.species_id] = per_species
                if not (per_species["gbif"]["available"] and per_species["ncbi"]["available"]):
                    degraded = True
                enriched.append(updated)
            else:  # a provider supplied by a test
                one, one_degraded = provider.enrich_all([candidate])
                enriched.extend(one)
                degraded = degraded or one_degraded

        updates: dict[str, Any] = {
            "candidates": enriched,
            "taxonomy_degraded": degraded,
            "taxonomy_report": report,
        }
        if degraded:
            # Named for the sources that actually ran. Calling a live GBIF
            # outage "a mocked taxonomy source" was false on every real request.
            mode = getattr(provider, "mode", None)
            which = {
                "mock": "A mocked taxonomy source",
                "real": "A live taxonomy source",
            }.get(mode, "A taxonomy source")
            updates["warnings"] = [
                *state.warnings,
                f"{which} was unavailable for at least one candidate; those candidates "
                "are reported as unverified and no identifier was inferred to fill the "
                "gap.",
            ]

        # Carry the annotations into the decision that was already made. The
        # identity of the primary species is looked up by species_id, so an
        # enrichment cannot substitute a different one for it.
        if decision is not None:
            by_id = {candidate.species_id: candidate for candidate in enriched}
            primary = (
                by_id.get(decision.primary_species.species_id)
                if decision.primary_species is not None
                else None
            )
            updates["decision"] = decision.model_copy(
                update={"candidates": enriched, "primary_species": primary}
            )
        tracer.record({
            "node": "taxonomy",
            "operation": "validate_taxonomy_with_mock_gbif_and_ncbi",
            "duration_ms": _ms_since(started),
            "taxonomy_provider_mode": getattr(provider, "mode", None),
            "taxonomy_available": not degraded,
        })
        return updates

    return _node


# --- 6. explain (LLM call 2 of 2) ------------------------------------------

def make_explain_node(
    llm: Any, config: RecognitionConfig, tracer: RecognitionTracer = _NULL_TRACER,
) -> Node:
    """Ask the model to phrase the evidence - then check what it wrote.

    An explanation naming a species the classifier did not return is discarded
    and the deterministic sentence is used instead. The model gets to phrase the
    finding; it never gets to add to it.
    """

    def _node(state: RecognitionState) -> dict:
        started = time.perf_counter()
        decision = state.decision
        assert decision is not None

        request = ExplainRequest(
            decision=decision.decision,
            text_alignment=decision.text_alignment,
            primary_species=(
                decision.primary_species.scientific_name if decision.primary_species else None
            ),
            candidate_names=tuple(c.scientific_name for c in decision.candidates),
            top_score=decision.candidates[0].classification_score if decision.candidates else None,
            margin=state.margin,
            taxonomy_status=(
                decision.primary_species.taxonomy_status if decision.primary_species else None
            ),
            recognition_mode=state.classification_mode or "unknown",
            classifier_version=state.classifier_version or "unknown",
            visual_evidence_sufficient=state.visual_evidence_sufficient,
        )

        text = None
        calls = 0
        # The remaining half of the two-call budget, and only that.
        remaining = config.reasoning_llm_max_calls_per_request - state.llm_plan_calls

        # A planner that failed forfeits the explanation call.
        #
        # Whatever went wrong - timeout, auth error, empty body, malformed JSON,
        # invalid schema, forbidden step - the provider has already shown it is
        # not answering to contract on this request. Spending the second call to
        # ask it again would be a retry wearing a different hat, and the
        # deterministic explanation is right there. So the explainer runs only
        # when the plan genuinely came from the model.
        planner_succeeded = state.plan_source == "llm"

        if (
            getattr(llm, "enabled", False)
            and hasattr(llm, "explain")
            and remaining > 0
            and planner_succeeded
        ):
            calls = 1
            try:
                produced = llm.explain(request)
            except Exception as exc:  # noqa: BLE001
                _logger.info(
                    "[Recognition] explainer raised (%s); deterministic explanation used.",
                    type(exc).__name__,
                )
                produced = None
            if produced is not None and explanation_is_grounded(produced, request):
                text = produced.strip()
            elif produced is not None:
                _logger.info("[Recognition] explanation rejected as ungrounded; using rules.")

        gbif_mode, ncbi_mode, taxonomy_executed = taxonomy_source_modes(state)

        source = "llm" if text else "deterministic"
        body = text or _deterministic_explanation(state, request)
        # Appended outside whatever the model wrote, and built from runtime
        # evidence rather than a fixed string: the disclosure is a fact about
        # this run, so neither the model nor a stale constant may author it.
        body = f"{body} {_safety_footer(state, gbif_mode, ncbi_mode, taxonomy_executed)}"

        tracer.record({
            "node": "explain",
            "operation": "explain",
            "duration_ms": _ms_since(started),
            "llm_role": "explainer",
            "llm_call_count": calls,
            "fallback": source != "llm",
        })
        return {
            "decision": decision.model_copy(update={"explanation": body}),
            "llm_explain_calls": calls,
            "reasoning_llm_calls": state.llm_plan_calls + calls,
            "reasoning_llm_used": state.reasoning_llm_used or source == "llm",
            "explanation_source": source,
        }

    return _node


def _classifier_phrase(state: RecognitionState) -> str:
    """How to name the thing that produced the labels, given what actually ran."""
    if state.classification_mode == RECOGNITION_MODE_MOCK_CLASSIFICATION:
        return "The Sprint 2 BioCLIP-2 classification mock"
    return "Remote BioCLIP-2 inference"


def _taxonomy_sentence(candidate, gbif_mode: str, ncbi_mode: str,
                       taxonomy_executed: bool) -> str:
    """What the taxonomy sources actually said about this candidate.

    Driven by the identifiers on the candidate rather than by the status label
    alone, so the sentence cannot contradict `gbif_id` and `ncbi_taxid` in the
    same response. This used to branch on `mock_verified` / `partial` / else;
    Phase 4 added the real-mode status `verified`, which matched neither, so a
    fully verified candidate fell through and was described as having no
    identifier at all - the exact opposite of the evidence beside it.
    """
    if not taxonomy_executed:
        return " No taxonomy lookup was performed for it."

    gbif_label = _source_label("GBIF", gbif_mode)
    ncbi_label = _source_label("NCBI", ncbi_mode)
    has_gbif = candidate.gbif_id is not None
    has_ncbi = candidate.ncbi_taxid is not None

    if has_gbif and has_ncbi:
        return (
            f" Both taxonomy sources supplied an identifier for it: {gbif_label} returned "
            f"{candidate.gbif_id} and {ncbi_label} returned taxid {candidate.ncbi_taxid}."
        )
    if has_gbif:
        return (
            f" {_source_label('GBIF', gbif_mode, sentence_start=True)} supplied identifier "
            f"{candidate.gbif_id}; {ncbi_label} supplied none, and its identifier is "
            "reported as null rather than filled in."
        )
    if has_ncbi:
        return (
            f" {_source_label('NCBI', ncbi_mode, sentence_start=True)} supplied taxid "
            f"{candidate.ncbi_taxid}; {gbif_label} supplied none, and its identifier is "
            "reported as null rather than filled in."
        )
    return (
        f" Neither {gbif_label} nor {ncbi_label} supplied an identifier for it; both are "
        "reported as null rather than filled in."
    )


def _deterministic_explanation(state: RecognitionState, request: ExplainRequest) -> str:
    """A grounded sentence built only from what the workflow actually computed."""
    gbif_mode, ncbi_mode, taxonomy_executed = taxonomy_source_modes(state)
    classifier = _classifier_phrase(state)

    if request.decision == "not_identified":
        body = (
            f"{classifier} returned no taxonomic label for this image confident enough "
            "to name a species."
        )
    else:
        score = "unknown" if request.top_score is None else round(request.top_score, 4)
        margin = "n/a" if request.margin is None else round(request.margin, 4)
        verb = "supports" if request.decision == "identified" else "does not conclusively support"
        top = state.candidates[0]
        body = (
            f"{classifier} {verb} {top.scientific_name} as the "
            f"highest-ranked taxonomic label (classification score {score}, margin over the "
            f"next label {margin})."
        )
        body += _taxonomy_sentence(top, gbif_mode, ncbi_mode, taxonomy_executed)

    if request.text_alignment == "agree":
        body += " The instruction names the same species as the classifier."
    elif request.text_alignment == "conflict":
        body += (
            " The instruction names a different species from the highest-ranked label, so "
            "the result was downgraded; the text cannot introduce a taxon the classifier "
            "did not return."
        )
    return body


def _safety_footer(state: RecognitionState, gbif_mode: str, ncbi_mode: str,
                   taxonomy_executed: bool) -> str:
    """Appended to every explanation, whoever wrote it.

    Bolting this on outside the model's text is deliberate: the provenance
    disclaimer is a fact about the system, so it must not depend on the model
    having remembered to include it - and the model may not author it either.

    It used to be a fixed string asserting a Sprint 2 mock and mocked taxonomy
    no matter what had run, so every live answer carried a disclaimer denying
    the very providers that produced it. Both halves are now chosen from the
    runtime evidence, and each describes only what is actually proven.
    """
    if state.classification_mode == RECOGNITION_MODE_MOCK_CLASSIFICATION:
        classification = (
            "Species classification is produced by a deterministic Sprint 2 mock of "
            "BioCLIP-2, not by real BioCLIP-2 inference; the classification score is a "
            "test value, not a probability."
        )
    else:
        classification = (
            "Species classification is produced by real remote BioCLIP-2 inference; the "
            "classification score is a ranking score over the model's label set, not a "
            "calibrated probability."
        )

    if not taxonomy_executed:
        taxonomy = (
            "No GBIF or NCBI lookup was performed for this request, so no taxonomic "
            "identifier is reported."
        )
    elif gbif_mode == "mock" and ncbi_mode == "mock":
        taxonomy = (
            "GBIF and NCBI validation are mocked and were not checked against the live "
            "databases."
        )
    elif gbif_mode == "real" and ncbi_mode == "real":
        taxonomy = (
            "GBIF and NCBI identifiers come from live lookups against the public GBIF and "
            "NCBI services; a source that was unavailable or did not match leaves its "
            "identifier null rather than filled in."
        )
    else:
        # Mixed or undeclared. Name each source separately rather than picking
        # one word that would be wrong about the other.
        taxonomy = (
            f"Taxonomy identifiers come from {_source_label('GBIF', gbif_mode)} and "
            f"{_source_label('NCBI', ncbi_mode)}; a source that was unavailable or did "
            "not match leaves its identifier null rather than filled in."
        )

    return f"{classification} {taxonomy}"


# --- 7. delegate_if_needed --------------------------------------------------

def make_delegation_node() -> Node:
    """Decide whether another agent's capability is needed - and only decide.

    Recognition never picks up the phone. It names a capability it cannot
    provide; the Capability Resolver chooses who actually provides it.
    """

    def _node(state: RecognitionState) -> dict:
        decision, evidence = state.decision, state.text_evidence
        assert decision is not None and evidence is not None

        if evidence.intent != "scientific_follow_up":
            return {}
        # Never delegate on an unresolved identification: the follow-up would be
        # asked about a species this agent did not actually establish.
        if decision.decision != "identified" or decision.primary_species is None:
            return {}

        capability = evidence.requested_capability
        if capability is None:
            return {}

        resume_key = HELPER_OUTPUT_KEYS.get(capability)
        if resume_key and resume_key in state.shared_context:
            _logger.info("[Recognition] resume: %r already present, completing", resume_key)
            return {}

        species = decision.primary_species.scientific_name
        _logger.info("[Recognition] needs_agent -> capability hint %s", capability)
        return {
            "delegate_to": capability,
            "delegation_prompt": (
                f"Recognition identified the species in the supplied image as {species}. "
                f"Provide the {capability.lower()} information the user asked for about "
                f"this species."
            ),
        }

    return _node
