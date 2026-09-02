"""
Probes every model NIM's catalog claims to support tool-calling for, with a
raw minimal HTTP request (no LangChain, no retry/fallback machinery), and
reports which ones are ACTUALLY live right now vs. which 410/404/error out
despite list_nim_models.py claiming they're reachable.

We've hit this exact "the static catalog listing lags real availability"
problem repeatedly -- guessing one replacement model at a time off that list
has cost multiple rounds. This gets a full, current picture in a single run.

Run from the trait_discovery_agent directory (so .env is picked up):

    docker compose run --rm trait-discovery-agent python probe_all_models.py

Respects NIM's ~40 RPM shared free-tier limit with a small delay between
calls -- this will take a few minutes for the full list, that's expected.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")
base_url = os.getenv("NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1"

if not api_key:
    raise SystemExit("Set NVIDIA_NIM_API_KEY (or NIM_API_KEY) first.")

# Every supports_tools=True model from the last list_nim_models.py run, plus
# everything currently in our FALLBACK_MODELS chain (workflows/llm/client.py)
# -- deduplicated. Edit this list directly if you want to probe something
# else specific.
MODELS = [
    "bytedance/seed-oss-36b-instruct",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-v3.2",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/gemma-4-31b-it",
    "ibm/granite-3.3-8b-instruct",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct",
    "meta/llama-3.3-70b-instruct",
    "microsoft/phi-4-mini-instruct",
    "minimaxai/minimax-m2",
    "mistralai/ministral-3-14b-instruct-2512",
    "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-nemotron",
    "mistralai/mistral-small-3.1-24b-instruct-2503",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2.6",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "nvidia/llama-3.1-nemotron-nano-4b-v1.1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwq-32b",
    "stepfun-ai/step-3.5-flash",
    "stepfun-ai/step-3.7-flash",
    "z-ai/glm-5.1",
    "z-ai/glm4.7",
    "z-ai/glm5",
]

url = base_url.rstrip("/") + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

alive = []
dead = []

for i, model in enumerate(MODELS, 1):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with just: OK"}],
        "max_tokens": 5,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"].get("content", "")
            print(f"[{i}/{len(MODELS)}] ALIVE   {model}  -> {content!r}")
            alive.append(model)
        else:
            body = resp.text[:150].replace("\n", " ")
            print(f"[{i}/{len(MODELS)}] DEAD    {model}  [{resp.status_code}] {body}")
            dead.append((model, resp.status_code, body))
    except Exception as e:
        print(f"[{i}/{len(MODELS)}] ERROR   {model}  {e!r}")
        dead.append((model, "exc", str(e)))
    time.sleep(1.6)  # ~37 RPM, under the shared ~40 RPM free-tier ceiling

print("\n" + "=" * 60)
print(f"ALIVE ({len(alive)}):")
for m in alive:
    print(" ", m)
print(f"\nDEAD/ERROR ({len(dead)}):")
for m, code, body in dead:
    print(f"  {m}  [{code}]")
