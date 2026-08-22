from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ...contracts import success
from ...core.dependencies import Identity, get_db, require_user, require_user_csrf
from ...schemas import ProfileUpdateRequest, user_view

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def me(request: Request, identity: Identity = Depends(require_user)) -> dict:
    return success(request, user_view(identity.user))


@router.patch("/me")
async def update_me(
    payload: ProfileUpdateRequest,
    request: Request,
    identity: Identity = Depends(require_user_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    profile = identity.user.profile
    if profile is not None:
        for field, value in payload.model_dump(exclude_unset=True).items():
            if field == "country" and value:
                value = value.upper()
            setattr(profile, field, value)
        await db.commit()
    return success(request, user_view(identity.user))
