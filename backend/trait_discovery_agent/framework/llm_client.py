import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

def get_nim_llm(temperature: float | None = None) -> ChatNVIDIA:


    """Shared NIM-backed chat model. LangGraph consumes this directly.
    CrewAI wraps it via LiteLLM's NVIDIA NIM provider string """



    api_key = os.environ["NVIDIA_NIM_API_KEY"]          # KeyError on purpose if missing — fail loud
    model = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct",)
    temp = temperature if temperature is not None else float(os.environ.get("NIM_TEMPERATURE", "0.2"))

    return ChatNVIDIA(model=model, api_key=api_key, temperature=temp,timeout=30, max_retries=2)