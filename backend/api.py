"""Compatibility entry point for the central Umbrella API.

Run from the repository root with:

    python -m uvicorn backend.api:app --reload --port 8000

Or from the backend directory with:

    python -m uvicorn backend.api:app --app-dir .. --reload --port 8000
"""

from backend.app.main import app, create_app

__all__ = ["app", "create_app"]
