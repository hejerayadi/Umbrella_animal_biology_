"""Platform fixes that must happen before an event loop exists.

One thing lives here, and it is not optional on Windows: psycopg's async driver
refuses to run on the default `ProactorEventLoop` with

    Psycopg cannot use the 'ProactorEventLoop' to run in async mode

Without the selector policy the agent still boots - it just cannot reach
Postgres, so `build_checkpointer` falls back to in-memory and CONTINUE stops
resuming. A silent loss of durability is exactly the failure worth spending a
module on.

Called from every entry point (`create_app`, `__main__`, the test conftest, the
scripts) rather than run as an import side effect, so it is visible where it
matters and cannot fire twice in surprising order.
"""
from __future__ import annotations

import asyncio
import sys


def use_selector_event_loop() -> bool:
    """Install the selector event loop policy on Windows.

    Returns whether the policy was changed. A no-op everywhere else: Linux and
    macOS already default to a selector loop, and forcing one there would only
    give up the faster default for nothing.

    Must run before the loop is created - after that, the policy is ignored.
    """
    if sys.platform != "win32":
        return False

    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:  # pragma: no cover - non-Windows
        return False

    if isinstance(asyncio.get_event_loop_policy(), policy):
        return False

    asyncio.set_event_loop_policy(policy())
    return True
