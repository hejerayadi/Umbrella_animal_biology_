"""
DNAModelRouter — cascade router for DNA gap prediction.

Implements the ordered fallback chain:
  DNABERT → NucleotideTransformer → [Seq2Seq] → AzureGPT (fallback)

Seq2Seq is skipped silently in Sprint 1 (``_seq2seq_available`` returns
``False``). Azure GPT-5.1 via ``get_llm()`` is the final fallback when all
DL models are exhausted or fail.

Requirements covered: 1.1, 1.2, 1.4, 1.5, 1.6, 5.3, 8.2
"""

from __future__ import annotations

import json
import logging

import torch
from langchain_core.messages import HumanMessage, SystemMessage

from backend.orchestrator.llm import get_llm

from ...schema import GapPrediction, GapRegion
from .dnabert_predictor import DNABERTPredictor
from .nucleotide_transformer_predictor import NucleotideTransformerPredictor
from .seq2seq_predictor import Seq2SeqPredictor
from .utils import FlanksWindow, _normalize_length, _resolve_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical model order
# ---------------------------------------------------------------------------

CANONICAL_ORDER: list[str] = ["DNABERT", "NucleotideTransformer", "Seq2Seq"]

# ---------------------------------------------------------------------------
# DNAModelRouter
# ---------------------------------------------------------------------------


class DNAModelRouter:
    """Singleton cascade router: DNABERT → NucleotideTransformer → [Seq2Seq] → AzureGPT.

    Usage
    -----
    >>> router = DNAModelRouter.get()
    >>> prediction = router.predict(flanks, gap, previous_attempts)

    The singleton is created once per process lifecycle.  Predictors are
    instantiated lazily on first use.

    Requirements
    ------------
    - 1.1  DNABERT attempted first when ``previous_attempts`` is empty.
    - 1.2  NucleotideTransformer attempted second.
    - 1.4  AzureGPT fallback when all DL models exhausted.
    - 1.5  Seq2Seq skipped in Sprint 1 (``_seq2seq_available`` → False).
    - 1.6  ``GapPrediction.model_used`` written for every prediction.
    - 5.3  OOM triggers cache clear + cascade continuation.
    - 8.2  WARNING logged on every model failure.
    """

    _instance: "DNAModelRouter | None" = None

    def __init__(self) -> None:
        self._dnabert: DNABERTPredictor | None = None
        self._nt: NucleotideTransformerPredictor | None = None
        self._seq2seq = None  # Post-Sprint 1
        self._device: str = _resolve_device()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "DNAModelRouter":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Sprint 1 availability guard
    # ------------------------------------------------------------------

    def _seq2seq_available(self) -> bool:
        """Return ``True`` only when a trained Seq2Seq checkpoint is present locally.

        Delegates to ``Seq2SeqPredictor._seq2seq_available()`` which checks
        for the existence of ``seq2seq_checkpoint.pt`` in the module directory.

        Returns
        -------
        bool
            ``True`` if a local checkpoint file is found; ``False`` otherwise.
        """
        return Seq2SeqPredictor._seq2seq_available()

    # ------------------------------------------------------------------
    # Lazy predictor instantiation
    # ------------------------------------------------------------------

    def _get_predictor(
        self, model_name: str
    ) -> DNABERTPredictor | NucleotideTransformerPredictor | Seq2SeqPredictor:
        """Return the cached predictor instance for *model_name*.

        Instantiates the predictor on first access (lazy singleton).

        Parameters
        ----------
        model_name:
            One of ``"DNABERT"``, ``"NucleotideTransformer"``, or ``"Seq2Seq"``.

        Returns
        -------
        DNABERTPredictor | NucleotideTransformerPredictor | Seq2SeqPredictor
            The appropriate predictor singleton.

        Raises
        ------
        ValueError
            If *model_name* is not a recognised DL model.
        """
        if model_name == "DNABERT":
            if self._dnabert is None:
                self._dnabert = DNABERTPredictor()
            return self._dnabert

        if model_name == "NucleotideTransformer":
            if self._nt is None:
                self._nt = NucleotideTransformerPredictor()
            return self._nt

        if model_name == "Seq2Seq":
            if self._seq2seq is None:
                self._seq2seq = Seq2SeqPredictor()
            return self._seq2seq

        raise ValueError(
            f"Unknown DL model name '{model_name}'. "
            f"Expected one of: 'DNABERT', 'NucleotideTransformer', 'Seq2Seq'."
        )

    # ------------------------------------------------------------------
    # Azure GPT fallback
    # ------------------------------------------------------------------

    def _azure_gpt_predict(
        self,
        flanks: FlanksWindow,
        gap: GapRegion,
    ) -> GapPrediction:
        """Predict gap sequence via Azure GPT-5.1 (final fallback).

        Adapts the prompt from ``agent.py::model_selector_predictor`` and
        reuses ``parse_llm_prediction``-style JSON parsing.

        Parameters
        ----------
        flanks:
            Flanking context window around the gap.
        gap:
            Gap region metadata (``gap.length`` drives expected output size).

        Returns
        -------
        GapPrediction
            Prediction with ``model_used="Azure-GPT-5.1"``.
        """
        logger.debug(
            "DNAModelRouter: routing gap idx=%d length=%d to AzureGPT fallback.",
            gap.start,
            gap.length,
        )

        prompt = (
            f"Please predict the missing DNA sequence (length {gap.length}).\n"
            f"Sequence type: {flanks.sequence_type}\n"
            f"Flanking sequence before gap: {flanks.left_context}\n"
            f"Flanking sequence after gap: {flanks.right_context}\n\n"
            "You MUST reply with strict JSON matching this format:\n"
            '{"predicted_sequence": "...", "confidence": 0.8}'
        )

        llm = get_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a sequence predictor for reconstructing genome gaps. "
                    "Respond only in strict JSON."
                )
            ),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        predicted_seq, confidence = self._parse_llm_prediction(
            response.content, gap.length
        )

        # Ensure length contract is satisfied (Req 6.5)
        predicted_seq = _normalize_length(predicted_seq, gap.length)

        return GapPrediction(
            gap=gap,
            predicted_sequence=predicted_seq,
            model_used="Azure-GPT-5.1",
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # LLM response parser (reused from agent.py logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_prediction(response: str, gap_length: int) -> tuple[str, float]:
        """Parse JSON from an LLM response, with a safe fallback.

        Mirrors ``parse_llm_prediction`` in ``agent.py``.

        Parameters
        ----------
        response:
            Raw text content from the LLM.
        gap_length:
            Expected gap length used for the ``'N'``-padded fallback.

        Returns
        -------
        tuple[str, float]
            ``(predicted_sequence, confidence)`` — confidence defaults to
            ``0.0`` when parsing fails.
        """
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx : end_idx + 1]
                data = json.loads(json_str)
                seq = data.get("predicted_sequence", "N" * gap_length)
                conf = float(data.get("confidence", 0.0))
                return seq, conf
        except Exception as exc:
            logger.warning(
                "DNAModelRouter: failed to parse LLM prediction for gap length %d: %s",
                gap_length,
                exc,
            )

        logger.warning(
            "DNAModelRouter: falling back to default 'N' prediction for gap length %d.",
            gap_length,
        )
        return "N" * gap_length, 0.0

    # ------------------------------------------------------------------
    # Main cascade entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        flanks: FlanksWindow,
        gap: GapRegion,
        previous_attempts: list[str],
    ) -> GapPrediction:
        """Run the cascade and return the first successful ``GapPrediction``.

        Iterates ``CANONICAL_ORDER``, skipping:
        - models already in ``previous_attempts``,
        - Seq2Seq when ``_seq2seq_available()`` is ``False`` (Sprint 1).

        On any exception from a predictor: log at WARNING, clear the CUDA
        cache, and continue to the next model (Req 5.3, 8.2).

        When all DL models are exhausted, delegates to ``_azure_gpt_predict``
        (Req 1.4, 1.5).

        Every returned ``GapPrediction.predicted_sequence`` is length-
        normalised to ``gap.length`` before return (Req 6.5).

        Parameters
        ----------
        flanks:
            Flanking context window.
        gap:
            Gap region to fill.
        previous_attempts:
            List of model names already tried for this gap in prior calls.

        Returns
        -------
        GapPrediction
            Filled gap with ``model_used`` and ``confidence`` populated.
        """
        for model_name in CANONICAL_ORDER:
            # Skip models already attempted for this gap
            if model_name in previous_attempts:
                continue

            # Sprint 1: silently skip Seq2Seq when unavailable (Req 1.5)
            if model_name == "Seq2Seq" and not self._seq2seq_available():
                logger.debug(
                    "DNAModelRouter: Seq2Seq not available (Sprint 1), skipping."
                )
                continue

            try:
                predictor = self._get_predictor(model_name)
                seq, conf = predictor.predict(flanks, gap)
                # Normalize length to guarantee contract (Req 6.5)
                seq = _normalize_length(seq, gap.length)
                logger.debug(
                    "DNAModelRouter: model=%s gap_idx=%d confidence=%.4f",
                    model_name,
                    gap.start,
                    conf,
                )
                return GapPrediction(
                    gap=gap,
                    predicted_sequence=seq,
                    model_used=model_name,
                    confidence=conf,
                )
            except Exception as exc:
                # Req 5.3, 8.2: log warning + clear GPU cache + continue cascade
                logger.warning(
                    "DNAModelRouter: model %s failed for gap idx %d: %s",
                    model_name,
                    gap.start,
                    exc,
                )
                torch.cuda.empty_cache()
                continue

        # All DL models exhausted → GPT fallback (Req 1.4)
        logger.debug(
            "DNAModelRouter: all DL models exhausted for gap idx=%d, "
            "routing to AzureGPT fallback.",
            gap.start,
        )
        return self._azure_gpt_predict(flanks, gap)
