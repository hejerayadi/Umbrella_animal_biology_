from fastapi import APIRouter

from backend.agents.Protein_visualization.app.api.v1.routes import analyses, health, knowledge, taxonomy

router = APIRouter()
router.include_router(health.router)
router.include_router(analyses.router)
router.include_router(knowledge.router)
router.include_router(taxonomy.router)
