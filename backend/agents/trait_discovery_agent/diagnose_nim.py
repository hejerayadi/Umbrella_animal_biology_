"""
Standalone NIM connectivity diagnostic -- bypasses ALL of trait_discovery_agent's
retry/fallback/tool-loop machinery to isolate whether the "[###] Unknown Error /
page not found" is a library, config, or account-level issue.

Run from the trait_discovery_agent directory (so .env is picked up), e.g.:

    python diagnose_nim.py

or inside the container:

    docker compose run --rm trait-discovery-agent python diagnose_nim.py
"""
import os
import traceback

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")
base_url = os.getenv("NIM_BASE_URL")  # None = public NVIDIA API catalog

print(f"API key present: {bool(api_key)} (starts with: {api_key[:8] + '...' if api_key else 'N/A'})")
print(f"NIM_BASE_URL: {base_url or '(unset -> public catalog default)'}")
print()

# 1) Raw HTTP request, no langchain at all -- rules out the SDK entirely.
import requests
url = (base_url or "https://integrate.api.nvidia.com/v1") + "/chat/completions"
payload = {
    "model": "deepseek-ai/deepseek-v3.2",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 10,
}
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
print(f"--- Raw HTTP POST to {url} ---")
try:
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    print("status_code:", resp.status_code)
    print("headers:", dict(resp.headers))
    print("body (first 2000 chars):", resp.text[:2000])
except Exception:
    print("Raw HTTP request itself raised:")
    traceback.print_exc()

print()

# 2) Same call through ChatNVIDIA -- if this fails identically, the raw HTTP
#    result above IS the real, untruncated error the pipeline is choking on.
print("--- Same call via ChatNVIDIA ---")
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=api_key, base_url=base_url, max_tokens=10)
    result = llm.invoke("Say OK.")
    print("SUCCESS:", result.content)
except Exception as e:
    print("ChatNVIDIA raised:")
    print(repr(e))
    traceback.print_exc()
