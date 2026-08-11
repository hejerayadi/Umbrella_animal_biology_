"""
Nucleotide Transformer zero-shot predictor for the dna_models module.

Implements a singleton predictor that uses InstaDeep's Nucleotide Transformer
family of checkpoints to fill genomic gaps via masked language modelling.

The checkpoint is automatically selected based on available VRAM:
  - >= 12 Go  → InstaDeepAI/nucleotide-transformer-2.5b-multi-species (~10 Go)
  - <  12 Go  → InstaDeepAI/nucleotide-transformer-500m-human-ref   (~2 Go)

Requirements covered: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 8.1, 8.4
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .utils import (
    FlanksWindow,
    _get_free_vram_gb,
    _normalize_length,
    _pick_dtype,
    _resolve_device,
    _symmetric_truncate,
)

if TYPE_CHECKING:
    from reconstruction_agent.schema import GapRegion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def _select_nt_checkpoint(vram_gb: float) -> str:
    """Select the Nucleotide Transformer checkpoint based on available VRAM.

    Parameters
    ----------
    vram_gb:
        Free VRAM in gigabytes (from ``_get_free_vram_gb()``).

    Returns
    -------
    str
        HuggingFace model identifier for the selected checkpoint.
    """
    if vram_gb >= 12.0:
        return "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    return "InstaDeepAI/nucleotide-transformer-500m-human-ref"


# ---------------------------------------------------------------------------
# Singleton predictor
# ---------------------------------------------------------------------------

class NucleotideTransformerPredictor:
    """Zero-shot gap predictor using InstaDeep's Nucleotide Transformer.

    Uses a singleton / class-level cache so that the model is loaded at most
    once per process lifecycle (Requirement 3.8).

    The predictor:
    1. Selects the checkpoint according to available VRAM (Req 3.1).
    2. Tokenises the flanks + gap region with the NT native tokenizer
       (6-nucleotide words, ``[MASK]`` tokens for the gap, Req 3.2).
    3. Symmetrically truncates if the total token count exceeds
       ``tokenizer.model_max_length`` (Req 3.3).
    4. Runs a forward pass (no_grad) and extracts logits at ``[MASK]``
       positions to decode the predicted sequence (Req 3.4).
    5. Computes ConfidenceScore as mean(max(softmax(logits[i])))
       over all ``[MASK]`` positions (Req 3.5).
    6. Normalises output length to ``gap.length`` exactly (Req 6.5).
    7. Handles ``torch.cuda.OutOfMemoryError`` by clearing the cache and
       re-raising so the router can cascade (Req 5.3).
    """

    # Class-level singletons (Req 3.8)
    _model: "AutoModelForMaskedLM | None" = None
    _tokenizer: "AutoTokenizer | None" = None
    _checkpoint: str | None = None  # tracks which checkpoint was loaded

    # VRAM budget used when picking dtype (keeps VRAM usage below total cap)
    _VRAM_BUDGET_GB: float = 6.0  # Req 3.6 — load in float16 if free < 6 Go

    @classmethod
    def _load(cls, device: str) -> None:
        """Load tokenizer and model if not already cached.

        Selects the checkpoint based on current free VRAM, picks the
        appropriate dtype, and calls ``model.eval()``.

        Parameters
        ----------
        device:
            Target device string (``"cuda"`` or ``"cpu"``).
        """
        if cls._model is not None:
            # Already loaded — singleton cache hit (Req 3.8)
            return

        vram_gb = _get_free_vram_gb()
        checkpoint = _select_nt_checkpoint(vram_gb)
        dtype = _pick_dtype(device, vram_budget_gb=cls._VRAM_BUDGET_GB)

        logger.debug(
            "NucleotideTransformerPredictor: loading checkpoint=%s device=%s dtype=%s "
            "free_vram_gb=%.2f",
            checkpoint,
            device,
            dtype,
            vram_gb,
        )

        cls._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        cls._model = AutoModelForMaskedLM.from_pretrained(
            checkpoint, torch_dtype=dtype
        ).to(device)
        cls._model.eval()
        cls._checkpoint = checkpoint

        logger.debug(
            "NucleotideTransformerPredictor: checkpoint loaded successfully (%s)",
            checkpoint,
        )

    def predict(
        self,
        flanks: FlanksWindow,
        gap: "GapRegion",
    ) -> tuple[str, float]:
        """Predict the sequence filling *gap* given the flanking context.

        Parameters
        ----------
        flanks:
            Left/right context window around the gap.
        gap:
            The gap region metadata (``gap.length`` drives mask count).

        Returns
        -------
        tuple[str, float]
            ``(predicted_sequence, confidence_score)`` where
            ``len(predicted_sequence) == gap.length`` and
            ``confidence_score ∈ [0.0, 1.0]``.

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Re-raised after ``empty_cache()`` so the router can cascade
            to the next model (Req 5.3).
        """
        device = _resolve_device()
        self._load(device)

        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        checkpoint = self.__class__._checkpoint

        # ----------------------------------------------------------------
        # Build input sequence: left + [MASK]×n + right
        # NT uses 6-nucleotide words — same sliding-window logic as DNABERT.
        # Corrected: n_mask = L-5 overlapping 6-mers, not round(L/6).
        # ----------------------------------------------------------------
        n_mask = max(1, gap.length - 5)
        mask_token: str = tokenizer.mask_token  # typically "[MASK]"

        left_seq: str = flanks.left_context
        right_seq: str = flanks.right_context
        mask_seq: str = " ".join([mask_token] * n_mask)

        # Tokenise each segment separately so we can truncate symmetrically
        left_tokens: list[int] = tokenizer.encode(
            left_seq, add_special_tokens=False
        )
        mask_tokens: list[int] = tokenizer.encode(
            mask_seq, add_special_tokens=False
        )
        right_tokens: list[int] = tokenizer.encode(
            right_seq, add_special_tokens=False
        )

        # ----------------------------------------------------------------
        # Symmetric truncation if total exceeds model_max_length (Req 3.3)
        # ----------------------------------------------------------------
        max_len: int = getattr(tokenizer, "model_max_length", 2048)
        total_tokens = len(left_tokens) + len(mask_tokens) + len(right_tokens) + 2  # +2 for special tokens
        if total_tokens > max_len:
            logger.debug(
                "NucleotideTransformerPredictor: truncating tokens (%d → %d) "
                "for gap idx %d (checkpoint=%s)",
                total_tokens,
                max_len,
                gap.start,
                checkpoint,
            )
            left_tokens, mask_tokens, right_tokens = _symmetric_truncate(
                left_tokens, mask_tokens, right_tokens, max_len
            )
            if len(left_tokens) == 0 and len(right_tokens) == 0:
                logger.warning(
                    "NucleotideTransformerPredictor | gap_start=%d gap_length=%d: "
                    "gap exceeds available context window (%d tokens), flanking "
                    "context reduced to 0 tokens. Predictions will have no flanking context.",
                    gap.start,
                    gap.length,
                    max_len,
                )

        # ----------------------------------------------------------------
        # Build final token ids tensor
        # ----------------------------------------------------------------
        cls_id: int = tokenizer.cls_token_id
        sep_id: int = tokenizer.sep_token_id

        all_token_ids: list[int] = (
            ([cls_id] if cls_id is not None else [])
            + left_tokens
            + mask_tokens
            + right_tokens
            + ([sep_id] if sep_id is not None else [])
        )

        input_ids = torch.tensor([all_token_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids)

        # Identify MASK token positions in the full sequence
        mask_token_id: int = tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        # ----------------------------------------------------------------
        # Forward pass (Req 3.4)
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        except torch.cuda.OutOfMemoryError as oom:
            logger.error(
                "NucleotideTransformerPredictor OOM for gap idx %d length %d: %s",
                gap.start,
                gap.length,
                oom,
            )
            torch.cuda.empty_cache()
            raise  # router will catch and cascade (Req 5.3)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

        # ----------------------------------------------------------------
        # Decode predicted sequence from [MASK] logits — majority vote
        # ----------------------------------------------------------------
        # Each MASK token at index idx (0-based) represents the 6-mer
        # starting at gap position idx, covering positions idx…idx+5.
        # We accumulate votes per gap position and resolve by majority.
        # Tie-breaking: token closest to the centre of the gap wins.
        #
        # NT's tokenizer.decode returns the full 6-mer string directly.
        centre_j = (len(mask_positions) - 1) / 2.0

        votes: dict[int, Counter] = {p: Counter() for p in range(gap.length)}
        centre_dist: dict[tuple[int, str], float] = {}

        for idx, pos in enumerate(mask_positions):
            pos_logits = logits[0, pos, :]  # (vocab_size,)
            probs = F.softmax(pos_logits, dim=-1)
            best_id = int(torch.argmax(probs).item())
            max_prob = float(probs[best_id].item())
            # NT tokenizer returns the 6-mer string (e.g. "ACGTAC")
            kmer = tokenizer.decode([best_id]).strip().upper().replace(" ", "")
            if not kmer:
                kmer = "N"

            for offset, base in enumerate(kmer):
                gap_pos = idx + offset
                if gap_pos >= gap.length:
                    break
                votes[gap_pos][base] += max_prob
                dist = abs(idx - centre_j)
                key = (gap_pos, base)
                if key not in centre_dist or dist < centre_dist[key]:
                    centre_dist[key] = dist

        predicted_tokens_list: list[str] = []
        mask_max_probs: list[float] = []

        for p in range(gap.length):
            pos_votes = votes[p]
            if not pos_votes:
                predicted_tokens_list.append("N")
                mask_max_probs.append(0.0)
                continue

            max_weight = max(pos_votes.values())
            candidates = [b for b, w in pos_votes.items() if w == max_weight]

            if len(candidates) == 1:
                winner = candidates[0]
            else:
                winner = min(
                    candidates,
                    key=lambda b: centre_dist.get((p, b), float("inf")),
                )

            predicted_tokens_list.append(winner)
            mask_max_probs.append(max_weight / max(1, sum(pos_votes.values())))

        # Join per-position winners into the raw sequence string
        raw_seq = "".join(predicted_tokens_list).upper()

        # ----------------------------------------------------------------
        # ConfidenceScore (Req 3.5) — mean of per-position max softmax prob
        # ----------------------------------------------------------------
        if mask_max_probs:
            confidence = float(sum(mask_max_probs) / len(mask_max_probs))
        else:
            confidence = 0.0

        # Clamp to [0.0, 1.0] for safety
        confidence = max(0.0, min(1.0, confidence))

        # ----------------------------------------------------------------
        # Normalise to gap.length exactly (Req 6.5)
        # ----------------------------------------------------------------
        predicted_sequence = _normalize_length(raw_seq, gap.length)

        # ----------------------------------------------------------------
        # Logging (Req 8.1, 8.4)
        # ----------------------------------------------------------------
        logger.debug(
            "NucleotideTransformerPredictor: checkpoint=%s gap_idx=%d gap_length=%d "
            "confidence=%.4f time_ms=%.1f",
            checkpoint,
            gap.start,
            gap.length,
            confidence,
            elapsed_ms,
        )

        return predicted_sequence, confidence
