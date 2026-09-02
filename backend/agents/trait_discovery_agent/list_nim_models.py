import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")
base_url = os.getenv("NIM_BASE_URL")  # None = public NVIDIA API catalog

if not api_key:
    raise SystemExit("Set NVIDIA_NIM_API_KEY (or NIM_API_KEY) first.")

kwargs = {"api_key": api_key}
if base_url:
    kwargs["base_url"] = base_url

models = ChatNVIDIA.get_available_models(**kwargs)

tool_models = sorted(m.id for m in models if getattr(m, "supports_tools", False))
all_models = sorted(m.id for m in models)

print(f"Endpoint: {base_url or 'public NVIDIA API catalog'}")
print(f"\n{len(all_models)} total models reachable with this key/endpoint.")
print(f"\n{len(tool_models)} of those report supports_tools=True:\n")
for m in tool_models:
    print(f"  {m}")