from __future__ import annotations

from typing import Any

import pytest

from mentat_session_logger.llm import OllamaClient, parse_json_response


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def test_ollama_client_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse({"response": "  ready  \n"})

    monkeypatch.setattr("mentat_session_logger.llm.requests.post", _fake_post)

    client = OllamaClient(
        endpoint="http://localhost:11434/api/generate",
        model="llama3.1:8b",
        timeout_seconds=30,
    )
    result = client.generate("ping")

    assert result == "ready"
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:11434/api/generate"
    assert calls[0]["json"] == {"model": "llama3.1:8b", "prompt": "ping", "stream": False}
    assert calls[0]["timeout"] == 30


def test_ollama_client_raises_when_response_string_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse({"not_response": "x"})

    monkeypatch.setattr("mentat_session_logger.llm.requests.post", _fake_post)

    client = OllamaClient(endpoint="http://example", model="model")
    with pytest.raises(ValueError, match="missing 'response' string"):
        client.generate("hello")


def test_parse_json_response_accepts_embedded_object() -> None:
    raw = 'Sure, here it is:\n{"primary_category":"IC_GAMEPLAY","summary":"ok"}\nThanks!'
    parsed = parse_json_response(raw)
    assert parsed["primary_category"] == "IC_GAMEPLAY"
    assert parsed["summary"] == "ok"


def test_parse_json_response_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        parse_json_response('["not","an","object"]')
