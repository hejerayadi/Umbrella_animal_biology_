"""Everything served under /api/v1.

One module that knows the v1 surface, so the application mounts a version
rather than a list of routers - and a future /api/v2 becomes a second include
rather than a rewrite. Both versions would reuse the same agent, services and
domain; only this transport layer is versioned.
"""

from __future__ import annotations

from fastapi import APIRouter

from reconstruction_agent.api.v1.endpoints import health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)

__all__ = ["api_router"]
