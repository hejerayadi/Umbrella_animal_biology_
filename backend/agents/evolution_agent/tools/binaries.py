"""Locate external bioinformatics executables.

Resolution order, applied identically to every tool:

  a. an explicit path in an environment variable (``MAFFT_BINARY``,
     ``IQTREE_BINARY``) — a bare command name is accepted and looked up on
     PATH;
  b. the first candidate command found on PATH;
  c. the legacy bundled path that used to be hard-coded in the wrappers.

Returning ``None`` means "not installed here": the caller turns that into
its own domain error, listing what was tried, so a missing binary is never
confused with a tool that ran and failed.
"""

from __future__ import annotations

import logging
import os
import shutil

_logger = logging.getLogger(__name__)

MAFFT_ENV_VAR = "MAFFT_BINARY"
IQTREE_ENV_VAR = "IQTREE_BINARY"

MAFFT_CANDIDATES = ("mafft.bat", "mafft")
IQTREE_CANDIDATES = ("iqtree3.exe", "iqtree3", "iqtree2.exe", "iqtree2", "iqtree")


def resolve_binary(
    env_var: str,
    candidates: tuple[str, ...],
    legacy_path: str | None = None,
) -> str | None:
    """Return an executable path, or ``None`` if nothing usable was found."""
    explicit = (os.environ.get(env_var) or "").strip().strip('"').strip("'")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        found = shutil.which(explicit)
        if found:
            return found
        # An explicit setting that does not resolve is a configuration error,
        # not a reason to silently fall back to something else.
        _logger.warning(
            "[binaries] %s is set but does not resolve to an executable: %r",
            env_var, explicit,
        )
        return None

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found

    if legacy_path and os.path.isfile(legacy_path):
        return legacy_path

    return None


def describe_search(
    env_var: str,
    candidates: tuple[str, ...],
    legacy_path: str | None = None,
) -> str:
    """Human-readable summary of where a binary was looked for."""
    parts = [
        f"${env_var}={os.environ.get(env_var) or '(unset)'}",
        f"PATH candidates: {', '.join(candidates)}",
    ]
    if legacy_path:
        parts.append(f"legacy path: {legacy_path}")
    return "; ".join(parts)
