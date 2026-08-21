"""
NIM (NVIDIA Inference Microservice) invocation layer, split by shape of the
call:

  client.py           - ChatNVIDIA client construction, error classifiers,
                         capacity-retry helper. Internal building blocks.
  json_completion.py  - invoke_with_fallback / invoke_json_with_fallback:
                         no tools bound, model just answers over fixed context.
  tool_loop.py         - invoke_tool_loop_with_fallback: bind_tools ReAct-style
                         loop (guide §2/§5), used by the subagents' own
                         multi-candidate decisions.

Everything below is re-exported here so existing callers can keep doing
`from workflows.llm import get_llm` / `invoke_with_fallback` / etc. without
caring which submodule actually defines it.
"""
from .client import (
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    FALLBACK_MODELS,
    MAX_CAPACITY_RETRIES,
    CAPACITY_RETRY_BASE_SECONDS,
    get_llm,
)
from .json_completion import invoke_with_fallback, invoke_json_with_fallback
from .tool_loop import invoke_tool_loop_with_fallback, MAX_TOOL_TURNS

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "FALLBACK_MODELS",
    "MAX_CAPACITY_RETRIES",
    "CAPACITY_RETRY_BASE_SECONDS",
    "MAX_TOOL_TURNS",
    "get_llm",
    "invoke_with_fallback",
    "invoke_json_with_fallback",
    "invoke_tool_loop_with_fallback",
]
