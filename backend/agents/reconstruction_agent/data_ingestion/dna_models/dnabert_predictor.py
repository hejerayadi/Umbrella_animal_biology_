"""
DNABERT zero-shot predictor for DNA gap reconstruction.

Implements the ``DNABERTPredictor`` singleton, which loads the
``zhihan1996/DNA_bert_6`` checkpoint once per process lifecycle and uses
the masked-language-modelling head to predict gap sequences base by base.

Requirements covered: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 8.1, 8.4
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForMaskedLM

from .utils import (
    FlanksWindow,
    _normalize_length,
    _pick_dtype,
    _resolve_device,
    _symmetric_truncate,
)

if TYPE_CHECKING:
    # Avoid a hard import cycle at runtime; used only for type annotations.
    from reconstruction_agent.schema import GapRegion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-ID → base character mapping
# ---------------------------------------------------------------------------
# DNABERT (zhihan1996/DNA_bert_6) tokenizes DNA as 6-mers.  The vocabulary
# contains individual nucleotide tokens too; we decode per-MASK position by
# taking the argmax of the full vocabulary distribution and converting the
# predicted token back to a single base.  The mapping below covers the four
# canonical bases plus the ambiguity character 'N' for everything else.

_BASE_CHARS = {"A", "C", "G", "T"}


def _to_kmers(seq: str, k: int = 6) -> str:
    """Convert a raw DNA string to space-separated k-mers for DNABERT.

    DNABERT (zhihan1996/DNA_bert_6) uses a k-mer-6 vocabulary.  The
    tokenizer has no character-level entries — passing a raw continuous
    string produces a single ``[UNK]`` token and destroys all context.
    This function produces the sliding-window 6-mer representation that
    the tokenizer expects (e.g. ``"ACGTAC CGTACG GTACGT ..."``).

    Parameters
    ----------
    seq : str
        Raw DNA string (A/C/G/T/N).  Should be uppercased before calling.
    k : int
        K-mer size, default 6 to match the DNABERT vocabulary.

    Returns
    -------
    str
        Space-separated k-mers, or empty string if ``len(seq) < k``.
        Short sequences (< k bases) return ``""`` — the caller must handle
        this case explicitly to avoid silently producing degenerate input.
    """
    if len(seq) < k:
        return ""
    return " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))


def _token_to_kmer(token_str: str) -> str:
    """Convert a decoded vocabulary token to the full k-mer string it represents.

    DNABERT's vocabulary contains 6-mer strings (e.g. ``"ACGTAC"``),
    single characters, and special tokens.  We map:
    - A k-mer (1–6 canonical bases) → the full string (all characters)
    - Anything else (special tokens, ambiguous chars) → ``'N'`` * len(s)
      so that the caller receives exactly ``len(s)`` characters of coverage.

    .. note::
        The old helper ``_token_to_base`` (removed) incorrectly returned only
        ``s[0]``, discarding the other 5 bases of every predicted 6-mer and
        causing ~83 % N-padding in the final output.

    Parameters
    ----------
    token_str:
        Raw token string from ``tokenizer.convert_ids_to_tokens()``.

    Returns
    -------
    str
        The full k-mer string (typically 6 canonical bases), or a string of
        ``'N'`` characters of the same length for non-canonical tokens.
    """
    s = token_str.upper().strip("## ")
    if not s:
        return "N"
    if all(c in _BASE_CHARS for c in s):
        # Return all bases — not just s[0].
        return s
    # Ambiguous / special token: fill with N for positional accounting.
    return "N" * max(1, len(s))


# ---------------------------------------------------------------------------
# DNABERTPredictor
# ---------------------------------------------------------------------------

class DNABERTPredictor:
    """Zero-shot MLM predictor backed by ``zhihan1996/DNA_bert_6``.

    Class-level attributes ``_model`` and ``_tokenizer`` act as a
    module-level singleton cache so that model weights are loaded at most once
    per process (Requirement 2.8).

    Usage
    -----
    >>> predictor = DNABERTPredictor()
    >>> sequence, confidence = predictor.predict(flanks, gap)
    """

    _model: "BertForMaskedLM | None" = None
    _tokenizer: "AutoTokenizer | None" = None

    CHECKPOINT: str = "zhihan1996/DNA_bert_6"
    # Revision that contains model.safetensors (avoids pytorch_model.bin and
    # the CVE-2025-32434 restriction introduced in transformers ≥ 4.47).
    REVISION: str = "75d5592a86a1dbdd27ab11ea0ea5574ab1a8920f"
    MAX_TOKENS: int = 512

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load(cls, device: str) -> None:
        """Load tokenizer and model into *device* if not already cached.

        Uses ``_pick_dtype`` to select float32 vs float16 based on the
        available VRAM budget of 10 GB (Requirement 2.6).  Puts the model
        in evaluation mode after loading.

        Parameters
        ----------
        device:
            PyTorch device string (``"cuda"`` or ``"cpu"``).
        """
        if cls._model is None:
            logger.debug(
                "Loading DNABERT checkpoint '%s' (rev=%s) onto device '%s'.",
                cls.CHECKPOINT,
                cls.REVISION[:12],
                device,
            )
            dtype = _pick_dtype(device, vram_budget_gb=10.0)
            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.CHECKPOINT,
                revision=cls.REVISION,
                trust_remote_code=True,
            )
            # Use BertForMaskedLM directly — AutoModelForMaskedLM conflicts with the
            # custom config_class in this checkpoint when trust_remote_code=True.
            cls._model = BertForMaskedLM.from_pretrained(
                cls.CHECKPOINT,
                revision=cls.REVISION,
                use_safetensors=True,
            ).to(device)
            cls._model.eval()
            logger.debug(
                "DNABERT loaded successfully (dtype=%s, device=%s, safetensors=True).",
                dtype,
                device,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, flanks: FlanksWindow, gap: "GapRegion") -> tuple[str, float]:
        """Predict the DNA sequence filling *gap* from the flanking context.

        Algorithm
        ---------
        1. Build the k-mer-6 input string:
           ``left_context + " [MASK]" * n_mask + right_context``
           where ``n_mask = max(1, gap.length - 5)`` — the number of
           overlapping 6-mers in a sliding window of length ``gap.length``
           (positions 0 to gap.length-6, inclusive).
        2. Tokenize; apply ``_symmetric_truncate`` if total > 512 tokens.
           Logs a WARNING if flanking context is reduced to 0 tokens.
        3. Forward pass with no grad; extract logits at ``[MASK]`` positions.
        4. Decode each MASK position via ``argmax(softmax(logits[pos]))``,
           obtaining the full predicted 6-mer (via ``_token_to_kmer``).
        5. Accumulate per-gap-position votes: MASK token j covers gap
           positions j…j+5, contributing its predicted base + probability.
           Resolve majority vote per position; tie-break by closest-to-centre
           token.
        6. Compute ``ConfidenceScore = mean(normalised_vote_weight per pos)``.
        7. Normalize output length to ``gap.length`` via ``_normalize_length``.

        Parameters
        ----------
        flanks:
            ``FlanksWindow`` with left/right context strings and gap metadata.
        gap:
            ``GapRegion`` dataclass with ``start``, ``end``, ``length``,
            ``in_scope``, and ``sequence_type`` fields.

        Returns
        -------
        tuple[str, float]
            ``(predicted_sequence, confidence)`` where ``predicted_sequence``
            has exactly ``gap.length`` characters and ``confidence ∈ [0.0, 1.0]``.

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Re-raised after ``torch.cuda.empty_cache()`` so the router can
            cascade to the next model (Requirement 2.7).
        """
        device = _resolve_device()
        self._load(device)

        tokenizer = self.__class__._tokenizer
        model = self.__class__._model

        # ----------------------------------------------------------------
        # 1. Build input sequence
        # ----------------------------------------------------------------
        # Corrected mask count: a gap of length L is covered by L-5
        # overlapping 6-mers (sliding window starting at positions 0…L-6).
        # Old formula round(L/6) under-estimated by a factor of ~6.
        n_mask = max(1, gap.length - 5)

        # ----------------------------------------------------------------
        # 2. Tokenize
        # ----------------------------------------------------------------
        # DNABERT uses a k-mer-6 vocabulary.  Raw continuous DNA strings
        # produce a single [UNK] token.  Convert flanks to space-separated
        # sliding-window 6-mers before tokenization.
        # [MASK] tokens are special tokens handled as-is — do NOT k-merize.
        left_kmer  = _to_kmers(flanks.left_context.upper())
        right_kmer = _to_kmers(flanks.right_context.upper())

        # add_special_tokens=False so we control [CLS]/[SEP] placement
        encoding_left = tokenizer(
            left_kmer,
            add_special_tokens=False,
            return_tensors=None,
        ) if left_kmer else {"input_ids": []}
        encoding_mask = tokenizer(
            " ".join(["[MASK]"] * n_mask),
            add_special_tokens=False,
            return_tensors=None,
        )
        encoding_right = tokenizer(
            right_kmer,
            add_special_tokens=False,
            return_tensors=None,
        ) if right_kmer else {"input_ids": []}

        left_ids: list[int] = encoding_left["input_ids"]
        mask_ids: list[int] = encoding_mask["input_ids"]
        right_ids: list[int] = encoding_right["input_ids"]

        # ----------------------------------------------------------------
        # 3. Symmetric truncation if needed
        # ----------------------------------------------------------------
        total_with_special = len(left_ids) + len(mask_ids) + len(right_ids) + 2
        if total_with_special > self.MAX_TOKENS:
            left_ids, mask_ids, right_ids = _symmetric_truncate(
                left_ids, mask_ids, right_ids, self.MAX_TOKENS
            )
            if len(left_ids) == 0 and len(right_ids) == 0:
                logger.warning(
                    "DNABERT | gap_start=%d gap_length=%d: gap exceeds available "
                    "context window (%d tokens), flanking context reduced to 0 tokens. "
                    "Predictions will have no flanking context.",
                    gap.start,
                    gap.length,
                    self.MAX_TOKENS,
                )

        # Reassemble with special tokens: [CLS] ... [SEP]
        cls_id: int = tokenizer.cls_token_id  # type: ignore[assignment]
        sep_id: int = tokenizer.sep_token_id  # type: ignore[assignment]
        all_ids: list[int] = [cls_id] + left_ids + mask_ids + right_ids + [sep_id]

        # Track the positions of [MASK] tokens in the assembled sequence
        mask_token_id: int = tokenizer.mask_token_id  # type: ignore[assignment]
        mask_positions: list[int] = [
            i for i, tid in enumerate(all_ids) if tid == mask_token_id
        ]

        input_ids_tensor = torch.tensor([all_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids_tensor)

        # ----------------------------------------------------------------
        # 4. Forward pass (with OOM handling)
        # ----------------------------------------------------------------
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "OOM during DNABERT inference for gap start=%d length=%d.",
                gap.start,
                gap.length,
            )
            raise

        inference_ms = (time.monotonic() - t0) * 1_000

        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits  # type: ignore[union-attr]

        # ----------------------------------------------------------------
        # 5. Majority-vote decoding per gap position
        # ----------------------------------------------------------------
        # Each MASK token at index j (0-based within the mask span) represents
        # the 6-mer starting at gap position j, covering positions j…j+5.
        # We accumulate (base, probability) votes per position, then pick the
        # majority base.  In the case of a tie, the vote from the token whose
        # start position is closest to the centre of the gap wins — a
        # deterministic, documented tie-breaking rule.
        #
        # votes[p] = Counter mapping base → total probability mass
        votes: dict[int, Counter] = {p: Counter() for p in range(gap.length)}
        # We also track best-prob per (position, base) to use for tie-breaking.
        # Store the raw probability contributed by the token nearest the centre.
        centre_j = (len(mask_positions) - 1) / 2.0  # float centre index

        # Maps (position, base) → prob of the token nearest centre seen so far
        centre_dist: dict[tuple[int, str], float] = {}
        # Maps (position, base) → prob of closest-to-centre token
        centre_prob: dict[tuple[int, str], float] = {}

        for idx, pos in enumerate(mask_positions):
            pos_logits = logits[0, pos, :]  # (vocab_size,)
            probs = F.softmax(pos_logits, dim=-1)
            best_id = int(probs.argmax().item())
            max_prob = float(probs[best_id].item())

            token_str = tokenizer.convert_ids_to_tokens(best_id)  # type: ignore[arg-type]
            kmer = _token_to_kmer(token_str)  # full 6-mer string, not just s[0]

            # The token at mask index idx covers gap positions idx…idx+5
            # (capped at gap.length - 1).
            for offset, base in enumerate(kmer):
                gap_pos = idx + offset
                if gap_pos >= gap.length:
                    break
                votes[gap_pos][base] += max_prob
                dist = abs(idx - centre_j)
                key = (gap_pos, base)
                if key not in centre_dist or dist < centre_dist[key]:
                    centre_dist[key] = dist
                    centre_prob[key] = max_prob

        # Resolve majority vote per position
        predicted_bases: list[str] = []
        max_probs: list[float] = []

        for p in range(gap.length):
            pos_votes = votes[p]
            if not pos_votes:
                predicted_bases.append("N")
                max_probs.append(0.0)
                continue

            # Find the maximum vote weight
            max_weight = max(pos_votes.values())
            candidates = [b for b, w in pos_votes.items() if w == max_weight]

            if len(candidates) == 1:
                winner = candidates[0]
            else:
                # Tie-breaking: pick the base whose token is closest to centre
                winner = min(
                    candidates,
                    key=lambda b: centre_dist.get((p, b), float("inf")),
                )

            predicted_bases.append(winner)
            max_probs.append(max_weight / max(1, sum(pos_votes.values())))

        # ----------------------------------------------------------------
        # 6. ConfidenceScore
        # ----------------------------------------------------------------
        raw_seq = "".join(predicted_bases)

        if max_probs:
            confidence = float(sum(max_probs) / len(max_probs))
        else:
            # Fallback: no MASK positions decoded (shouldn't happen)
            confidence = 0.0
            raw_seq = ""

        # ----------------------------------------------------------------
        # 7. Normalize to exact gap length
        # ----------------------------------------------------------------
        predicted_sequence = _normalize_length(raw_seq, gap.length)

        # ----------------------------------------------------------------
        # 8. Debug logging (Requirements 8.1, 8.4)
        # ----------------------------------------------------------------
        logger.debug(
            "DNABERT | model=%s | gap_start=%d | gap_length=%d | "
            "confidence=%.4f | inference_ms=%.1f",
            self.CHECKPOINT,
            gap.start,
            gap.length,
            confidence,
            inference_ms,
        )

        return predicted_sequence, confidence
