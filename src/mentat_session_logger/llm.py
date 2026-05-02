from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests


class LlmClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class OllamaClient:
    endpoint: str
    model: str
    timeout_seconds: int = 120

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid LLM response")
        result = payload.get("response")
        if not isinstance(result, str):
            raise ValueError("LLM response missing 'response' string")
        return result.strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parsed = _parse_embedded_json_object(raw, exc)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON must be an object")
    return cast(dict[str, Any], parsed)


def _parse_embedded_json_object(raw: str, original_error: json.JSONDecodeError) -> object:
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(raw[start:])
            return parsed
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
    raise ValueError(f"Invalid JSON from LLM: {original_error}") from original_error
