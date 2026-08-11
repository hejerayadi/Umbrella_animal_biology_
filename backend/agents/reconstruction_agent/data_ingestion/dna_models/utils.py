"""
Shared utilities for the dna_models module.

Provides device detection, VRAM management, sequence normalization,
symmetric truncation, and the FlanksWindow dataclass.

Requirements covered: 5.1, 5.2, 5.4, 5.5, 6.5, 2.3, 3.3, 10.2
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """Return the device string to use for model loading.

    Reads the ``DNA_MODEL_DEVICE`` environment variable (default ``"auto"``).
    When set to ``"auto"``, selects CUDA if available, otherwise CPU.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    env = os.getenv("DNA_MODEL_DEVICE", "auto").lower()
    if env == "auto":
        if torch.cuda.is_available():
            return "cuda"
        logger.info(
            "DNA_MODEL_DEVICE=auto: no CUDA GPU detected, falling back to CPU (float32)."
        )
        return "cpu"
    # Explicit override: "cuda" or "cpu"
    return env


# ---------------------------------------------------------------------------
# VRAM detection
# ---------------------------------------------------------------------------

def _get_free_vram_gb() -> float:
    """Return free VRAM in gigabytes for the default CUDA device.

    Uses ``torch.cuda.get_device_properties()`` to obtain total memory and
    ``torch.cuda.memory_reserved()`` to estimate the currently occupied
    portion.

    Returns
    -------
    float
        Free VRAM in GB.  Returns ``0.0`` when CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    total_bytes: int = props.total_memory
    reserved_bytes: int = torch.cuda.memory_reserved(0)
    free_bytes = max(0, total_bytes - reserved_bytes)
    return free_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# dtype selection
# ---------------------------------------------------------------------------

def _pick_dtype(device: str, vram_budget_gb: float) -> torch.dtype:
    """Choose a numerical precision based on device and VRAM budget.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    vram_budget_gb:
        The VRAM threshold (in GB) above which ``float32`` is preferred.
        If free VRAM is below this threshold, ``float16`` is used.

    Returns
    -------
    torch.dtype
        ``torch.float32`` on CPU or when VRAM is sufficient;
        ``torch.float16`` when VRAM is below the budget threshold.
    """
    if device == "cpu":
        return torch.float32
    free_gb = _get_free_vram_gb()
    if free_gb >= vram_budget_gb:
        return torch.float32
    return torch.float16


# ---------------------------------------------------------------------------
# Sequence length normalisation
# ---------------------------------------------------------------------------

def _normalize_length(seq: str, target_length: int) -> str:
    """Truncate or pad *seq* with ``'N'`` to match *target_length* exactly.

    Parameters
    ----------
    seq:
        Predicted DNA sequence string.
    target_length:
        Exact number of bases required (typically ``gap.length``).

    Returns
    -------
    str
        A string of exactly *target_length* characters.
    """
    if len(seq) > target_length:
        return seq[:target_length]
    if len(seq) < target_length:
        return seq + "N" * (target_length - len(seq))
    return seq


# ---------------------------------------------------------------------------
# Symmetric truncation
# ---------------------------------------------------------------------------

def _symmetric_truncate(
    left_tokens: list,
    mask_tokens: list,
    right_tokens: list,
    max_tokens: int,
) -> tuple[list, list, list]:
    """Symmetrically truncate flanks so the total fits within *max_tokens*.

    Only the left and right flanks are trimmed; ``mask_tokens`` (representing
    the gap region) are **never** modified.

    Special tokens (``[CLS]`` + ``[SEP]``) consume 2 slots from *max_tokens*.

    Parameters
    ----------
    left_tokens:
        Token IDs (or token strings) for the left context flank.
    mask_tokens:
        Token IDs representing the masked gap region.
    right_tokens:
        Token IDs for the right context flank.
    max_tokens:
        Maximum number of tokens the model accepts (e.g. 512 for DNABERT).

    Returns
    -------
    tuple[list, list, list]
        ``(left_trimmed, mask_tokens, right_trimmed)`` where flanks have been
        trimmed symmetrically if necessary.

    Raises
    ------
    ValueError
        When the gap alone exceeds the model context window (after subtracting
        the 2 special-token slots).
    """
    special_budget = 2  # [CLS] + [SEP]
    available = max_tokens - special_budget - len(mask_tokens)
    if available <= 0:
        raise ValueError(
            f"Gap region ({len(mask_tokens)} tokens) is too large for model "
            f"context window ({max_tokens} tokens). Cannot truncate flanks."
        )
    half = available // 2
    left_trimmed = left_tokens[-half:] if len(left_tokens) > half else left_tokens
    right_trimmed = right_tokens[:half] if len(right_tokens) > half else right_tokens
    return left_trimmed, mask_tokens, right_trimmed


# ---------------------------------------------------------------------------
# FlanksWindow dataclass
# ---------------------------------------------------------------------------

@dataclass
class FlanksWindow:
    """Context window around a gap, used as input to all DL predictors.

    Attributes
    ----------
    left_context:
        Left flanking DNA sequence (≤ 1 000 bp).
    gap_length:
        Exact number of bases in the gap (``gap.length``).
    right_context:
        Right flanking DNA sequence (≤ 1 000 bp).
    sequence_type:
        Propagated from ``GapRegion.sequence_type``; does **not** modify
        sequence content (``"nuclear"`` or ``"mitochondrial"``).
    """

    left_context: str
    gap_length: int
    right_context: str
    sequence_type: str  # "nuclear" | "mitochondrial"

    @property
    def total_length(self) -> int:
        """Total window length: left context + gap + right context."""
        return len(self.left_context) + self.gap_length + len(self.right_context)
