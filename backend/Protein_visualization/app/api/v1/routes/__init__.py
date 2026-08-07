from fastapi import APIRouter

from app.api.v1.routes import analyses, health, knowledge

router = APIRouter()
router.include_router(health.router)
router.include_router(analyses.router)
router.include_router(knowledge.router)
