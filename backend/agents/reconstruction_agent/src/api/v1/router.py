"""Aggregates the v1 routes under a single prefix.

One place that knows the v1 surface, so `api/app.py` mounts a version rather
than a list of routers - and a future v2 is a second include, not a rewrite.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.v1.routes import health, reconstruction

#: Everything served under /api/v1, in the `{data, meta, error}` envelope.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(reconstruction.router)

#: The orchestrator's endpoint, mounted at the root and NOT enveloped.
#: See `routes/reconstruction.py` for why it is separate.
orchestrator_router = reconstruction.orchestrator_router

__all__ = ["api_router", "orchestrator_router"]
