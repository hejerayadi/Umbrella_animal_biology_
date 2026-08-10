"""FLUX.2-pro image generation via Azure AI Foundry BFL provider API."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR / ".env", override=True)


class FluxGenerationError(RuntimeError):
    """Raised when FLUX.2-pro image generation fails."""


class FluxClient:
    """Client for FLUX.2-pro text-to-image generation on Azure Foundry."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        output_format: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint = self._normalize_endpoint(endpoint or os.getenv("AZURE_FLUX_ENDPOINT") or "")
        self.api_key = api_key or os.getenv("AZURE_FLUX_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_FLUX_API_VERSION", "preview")
        self.model = model or os.getenv("AZURE_FLUX_DEPLOYMENT", "FLUX.2-pro")
        self.width = width or int(os.getenv("AZURE_FLUX_WIDTH", "1024"))
        self.height = height or int(os.getenv("AZURE_FLUX_HEIGHT", "1024"))
        self.output_format = output_format or os.getenv("AZURE_FLUX_OUTPUT_FORMAT", "jpeg")
        self.timeout = timeout or int(os.getenv("AZURE_FLUX_TIMEOUT", "300"))
        self._session = session or requests.Session()

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint:
            return endpoint

        normalized_path = "/providers/blackforestlabs/v1/flux-2-pro"
        if endpoint.lower().endswith(normalized_path):
            return endpoint
        return f"{endpoint}{normalized_path}"

    def generate_image(self, prompt: str) -> str:
        """Generate an image and return a URL or base64 data URI."""
        if not self.endpoint:
            raise FluxGenerationError(
                "AZURE_FLUX_ENDPOINT is not set. Deploy FLUX.2-pro in Azure Foundry "
                "and copy the BFL provider endpoint to .env."
            )
        if not self.api_key:
            raise FluxGenerationError(
                "AZURE_FLUX_API_KEY (or AZURE_OPENAI_API_KEY) is not set."
            )
        if not prompt.strip():
            raise FluxGenerationError("Cannot generate an image from an empty prompt.")

        url = self.endpoint
        params = {"api-version": self.api_version}
        payload = {
            "model": self.model,
            "prompt": prompt.strip(),
            "width": self.width,
            "height": self.height,
            "output_format": self.output_format,
            "num_images": 1,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = self._session.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if exc.response is not None:
                detail = f" Response: {exc.response.text[:500]}"
            raise FluxGenerationError(f"FLUX.2-pro request failed: {exc}.{detail}") from exc

        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            encoded = base64.b64encode(response.content).decode("ascii")
            mime = content_type.split(";")[0].strip() or f"image/{self.output_format}"
            return f"data:{mime};base64,{encoded}"

        try:
            result: dict[str, Any] = response.json()
        except ValueError as exc:
            raise FluxGenerationError("FLUX.2-pro returned invalid JSON.") from exc

        return self._extract_image(result)

    def _extract_image(self, result: dict[str, Any]) -> str:
        # Direct URL fields
        for key in ("url", "image_url", "sample", "href"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        data = result.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("url", "b64_json", "image", "content"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        if key == "b64_json":
                            return f"data:image/{self.output_format};base64,{value.strip()}"
                        return value.strip()
            elif isinstance(first, str) and first.strip():
                return first.strip()

        for key in ("image", "b64_json", "output", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                if key == "b64_json":
                    return f"data:image/{self.output_format};base64,{value.strip()}"
                return value.strip()

        images = result.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                url = first.get("url") or first.get("b64_json")
                if isinstance(url, str):
                    if "b64_json" in first and first["b64_json"] == url:
                        return f"data:image/{self.output_format};base64,{url.strip()}"
                    return url.strip()

        raise FluxGenerationError(
            f"FLUX.2-pro response did not contain an image reference: {result!r}"
        )
