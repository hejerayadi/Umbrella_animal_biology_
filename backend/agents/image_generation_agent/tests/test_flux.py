from __future__ import annotations

import pytest
import requests

from backend.agents.Protein_visualization.flux_client import FluxClient, FluxGenerationError


class FakeResponse:
    def __init__(self, status_code: int, json_data=None, content: bytes = b"", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.headers = headers or {}
        self.text = content.decode("utf-8") if content else ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON")
        return self._json_data


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.last_request = None

    def post(self, url, params=None, headers=None, json=None, timeout=None):
        self.last_request = {
            "url": url,
            "params": params,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return self.response


def test_generate_image_returns_url_from_json():
    session = FakeSession(
        FakeResponse(
            200,
            json_data={"data": [{"url": "https://cdn.example/flux.jpg"}]},
            headers={"Content-Type": "application/json"},
        )
    )
    client = FluxClient(
        endpoint="https://resource.api.cognitive.microsoft.com",
        api_key="test-key",
        session=session,
    )

    image = client.generate_image("A scientific protein diagram")

    assert image == "https://cdn.example/flux.jpg"
    assert session.last_request is not None
    assert "flux-2-pro" in session.last_request["url"]
    assert session.last_request["json"]["model"] == "FLUX.2-pro"
    assert session.last_request["json"]["prompt"] == "A scientific protein diagram"
    assert session.last_request["headers"]["Authorization"] == "Bearer test-key"


def test_generate_image_returns_base64_from_b64_json():
    session = FakeSession(
        FakeResponse(
            200,
            json_data={"data": [{"b64_json": "abc123"}]},
            headers={"Content-Type": "application/json"},
        )
    )
    client = FluxClient(
        endpoint="https://resource.api.cognitive.microsoft.com",
        api_key="test-key",
        session=session,
    )

    image = client.generate_image("Prompt")

    assert image == "data:image/jpeg;base64,abc123"


def test_generate_image_raises_when_endpoint_missing():
    client = FluxClient(endpoint="", api_key="test-key")

    with pytest.raises(FluxGenerationError, match="AZURE_FLUX_ENDPOINT"):
        client.generate_image("Prompt")


def test_generate_image_raises_on_http_error():
    session = FakeSession(
        FakeResponse(
            401,
            json_data={"error": "Unauthorized"},
            headers={"Content-Type": "application/json"},
        )
    )
    client = FluxClient(
        endpoint="https://resource.api.cognitive.microsoft.com",
        api_key="bad-key",
        session=session,
    )

    with pytest.raises(FluxGenerationError, match="FLUX.2-pro request failed"):
        client.generate_image("Prompt")
