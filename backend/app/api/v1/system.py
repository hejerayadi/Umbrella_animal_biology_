from fastapi import APIRouter, Depends, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse

from ...contracts import success
from ...core.dependencies import Identity, require_admin, require_user

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request, _: Identity = Depends(require_user)) -> dict:
    return success(request, {"status": "ok"})


@router.get("/admin/docs", include_in_schema=False)
async def protected_docs(_: Identity = Depends(require_admin)) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/api/v1/admin/openapi.json", title="Umbrella API documentation"
    )


@router.get("/admin/openapi.json", include_in_schema=False)
async def protected_openapi(
    request: Request, _: Identity = Depends(require_admin)
) -> JSONResponse:
    return JSONResponse(request.app.openapi())

