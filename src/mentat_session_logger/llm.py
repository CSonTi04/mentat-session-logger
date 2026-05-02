from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

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


def parse_json_response(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON must be an object")
    return parsed
