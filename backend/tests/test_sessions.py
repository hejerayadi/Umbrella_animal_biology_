from unittest.mock import AsyncMock

import pytest

from backend.app.core.config import Settings
from backend.app.core.sessions import SessionStore


@pytest.mark.asyncio
async def test_reset_rate_limit_deletes_only_the_selected_bucket() -> None:
    redis = AsyncMock()
    store = SessionStore(redis, Settings(_env_file=None))

    await store.reset_rate_limit("login:127.0.0.1:account-hash")

    redis.delete.assert_awaited_once_with("auth:rate:login:127.0.0.1:account-hash")
