from __future__ import annotations

import requests

from backend.agents.image_generation_agent.tools.web_search import call_web_search


class FakeResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.last_request = None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last_request = {"url": url, "headers": headers, "json": json}
        return self.response


def test_call_web_search_success():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "organic": [
                    {
                        "title": "Insulin structure review",
                        "link": "https://example.com/insulin",
                        "snippet": "Human insulin folds into a compact hormone.",
                    }
                ]
            },
        )
    )

    result = call_web_search("human insulin structure", session=session, api_key="test-key")

    assert result.success is True
    assert len(result.results) == 1
    assert result.results[0].title == "Insulin structure review"
    assert result.results[0].link == "https://example.com/insulin"
    assert "compact hormone" in result.results[0].snippet
    assert session.last_request["headers"]["X-API-KEY"] == "test-key"


def test_call_web_search_missing_key():
    result = call_web_search("human insulin", api_key="")

    assert result.success is False
    assert result.error == "SERPER_API_KEY is not configured."


def test_call_web_search_api_failure():
    session = FakeSession(FakeResponse(500, {"error": "server error"}))

    result = call_web_search("human insulin", session=session, api_key="test-key")

    assert result.success is False
    assert "Web search request failed" in (result.error or "")


def test_call_web_search_parsing_skips_incomplete_rows():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "organic": [
                    {"title": "Valid", "link": "https://example.com/a", "snippet": "A"},
                    {"title": "", "link": "https://example.com/b", "snippet": "B"},
                    {"title": "No link", "snippet": "C"},
                ]
            },
        )
    )

    result = call_web_search("query", session=session, api_key="test-key")

    assert result.success is True
    assert len(result.results) == 1
