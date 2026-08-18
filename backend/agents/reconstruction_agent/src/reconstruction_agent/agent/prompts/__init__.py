"""LLM prompts, kept as reviewable artefacts rather than inline strings.

A prompt change alters agent behaviour as much as a code change does, so they
live in version control as their own files.
"""
from . import critic, explanation, planner

__all__ = ["critic", "explanation", "planner"]
