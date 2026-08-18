from fastapi import APIRouter

from . import admin, auth, system, users, workspace

router = APIRouter()
router.include_router(auth.router)
router.include_router(users.router)
router.include_router(admin.router)
router.include_router(workspace.router)
router.include_router(system.router)

