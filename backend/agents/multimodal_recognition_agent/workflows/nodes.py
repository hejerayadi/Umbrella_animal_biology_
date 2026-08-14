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
"""
from __future__ import annotations

import logging
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
from ..config import RecognitionConfig
from ..domain import confidence, ranking
from ..domain.errors import RecognitionError
from ..domain.models import RecognitionDecision
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


def _failed(exc: RecognitionError) -> dict:
    return {"error_code": exc.code.value, "error_message": exc.message}


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
    analyzer: RuleBasedTextAnalyzer, llm: Any, config: RecognitionConfig
) -> Node:
    """Deterministic evidence first, then let the model plan on top of it.

    The rules always run: they produce the evidence the planner is given, and
    they are the plan used whenever the model is absent, fails, or answers
    off-contract. The model can choose which optional steps run and refine the
    hints - it cannot skip a mandatory step or invent a route.
    """

    def _node(state: RecognitionState) -> dict:
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
        return updates

    return _node


# --- 3. classify_with_mock_bioclip2 ----------------------------------------

def make_classify_node(classifier: BioCLIP2Classifier, config: RecognitionConfig) -> Node:
    """The ONLY source of species candidates in this agent.

    One image goes in, an already-ranked list of taxonomic labels comes back.
    Nothing else in the workflow may add to that list, and there is no second
    path - no vector search, no reference lookup - through which a taxon could
    arrive.
    """

    def _node(state: RecognitionState) -> dict:
        assert state.normalized is not None
        # A plan may narrow K, never widen it past the configured maximum.
        planned = getattr(state.plan, "top_k", None) or config.top_k_species
        top_k = min(planned, config.top_k_species)

        try:
            predictions = classifier.classify(state.normalized, top_k)
            candidates = ranking.build_candidates(list(predictions), top_k=top_k)
        except RecognitionError as exc:
            return _failed(exc)

        _logger.info(
            "[Recognition] classified provider=%s mode=%s labels=%d top_k=%d",
            getattr(classifier, "provider_name", "unknown"),
            getattr(classifier, "recognition_mode", "unknown"),
            len(candidates), top_k,
        )
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

def make_taxonomy_node(provider: MockTaxonomyProvider) -> Node:
    """Validate and normalise through both mocked sources.

    Deliberately downstream of the confidence gate: neither source can pick a
    species, change a score, change the order or change the decision. They may
    only annotate candidates the classifier already produced. A missing
    identifier stays null - never invented.
    """

    def _node(state: RecognitionState) -> dict:
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
            updates["warnings"] = [
                *state.warnings,
                "A mocked taxonomy source was unavailable for at least one candidate; "
                "those candidates are reported as unverified and no identifier was "
                "inferred to fill the gap.",
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
        return updates

    return _node


# --- 6. explain (LLM call 2 of 2) ------------------------------------------

def make_explain_node(llm: Any, config: RecognitionConfig) -> Node:
    """Ask the model to phrase the evidence - then check what it wrote.

    An explanation naming a species the classifier did not return is discarded
    and the deterministic sentence is used instead. The model gets to phrase the
    finding; it never gets to add to it.
    """

    def _node(state: RecognitionState) -> dict:
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

        source = "llm" if text else "deterministic"
        body = text or _deterministic_explanation(state, request)
        body = f"{body} {_safety_footer()}"

        return {
            "decision": decision.model_copy(update={"explanation": body}),
            "llm_explain_calls": calls,
            "reasoning_llm_calls": state.llm_plan_calls + calls,
            "reasoning_llm_used": state.reasoning_llm_used or source == "llm",
            "explanation_source": source,
        }

    return _node


def _deterministic_explanation(state: RecognitionState, request: ExplainRequest) -> str:
    """A grounded sentence built only from what the workflow actually computed."""
    if request.decision == "not_identified":
        body = (
            "The Sprint 2 BioCLIP-2 classification mock returned no taxonomic label for "
            "this image confident enough to name a species."
        )
    else:
        score = "unknown" if request.top_score is None else round(request.top_score, 4)
        margin = "n/a" if request.margin is None else round(request.margin, 4)
        verb = "supports" if request.decision == "identified" else "does not conclusively support"
        top = state.candidates[0]
        body = (
            f"The Sprint 2 BioCLIP-2 classification mock {verb} {top.scientific_name} as the "
            f"highest-ranked taxonomic label (classification score {score}, margin over the "
            f"next label {margin})."
        )
        if top.taxonomy_status == "mock_verified":
            body += " Both mocked taxonomy sources held a record for it."
        elif top.taxonomy_status == "partial":
            body += (
                " Only one of the two mocked taxonomy sources supplied an identifier; the "
                "other is reported as null rather than filled in."
            )
        else:
            body += " Neither mocked taxonomy source supplied an identifier for it."

    if request.text_alignment == "agree":
        body += " The instruction names the same species as the classifier."
    elif request.text_alignment == "conflict":
        body += (
            " The instruction names a different species from the highest-ranked label, so "
            "the result was downgraded; the text cannot introduce a taxon the classifier "
            "did not return."
        )
    return body


def _safety_footer() -> str:
    """Appended to every explanation, whoever wrote it.

    Bolting this on outside the model's text is deliberate: the provenance
    disclaimer is a fact about the system, so it must not depend on the model
    having remembered to include it.
    """
    return (
        "Species classification is produced by a deterministic Sprint 2 mock of BioCLIP-2, "
        "not by real BioCLIP-2 inference; the classification score is a test value, not a "
        "probability. GBIF and NCBI validation are mocked and were not checked against the "
        "live databases."
    )


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
