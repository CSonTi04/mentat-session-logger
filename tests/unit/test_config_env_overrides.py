from __future__ import annotations

from pathlib import Path

import pytest

from mentat_session_logger.config import apply_env_overrides
from mentat_session_logger.models import EnvironmentConfig


def _base_cfg() -> EnvironmentConfig:
    root = Path("/tmp/env")
    return EnvironmentConfig(
        name="test",
        root=root,
        campaign_context_dir=root / "campaign_context",
        voiceprints_dir=root / "voiceprints",
        sessions_dir=root / "sessions",
        profiles_dir=root / "profiles",
    )


def test_no_env_vars_leaves_defaults_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MSL_LLM_PROVIDER",
        "MSL_LLM_ENDPOINT",
        "MSL_LLM_MODEL",
        "MSL_OUTPUT_LANGUAGE",
        "MSL_ASR_LANGUAGE",
        "MSL_DEFAULT_PIPELINE",
        "MSL_MIN_SPEAKERS",
        "MSL_MAX_SPEAKERS",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = apply_env_overrides(_base_cfg())

    assert cfg.llm_endpoint == "http://localhost:11434/api/generate"
    assert cfg.llm_model == "llama3.1:8b"
    assert cfg.llm_provider == "ollama"
    assert cfg.asr_language == "hu"
    assert cfg.output_language == "hu"
    assert cfg.min_speakers == 4
    assert cfg.max_speakers == 8
    assert cfg.default_pipeline == "default"


def test_string_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSL_LLM_ENDPOINT", "http://gpu-box:11434/api/generate")
    monkeypatch.setenv("MSL_LLM_MODEL", "mistral:7b")
    monkeypatch.setenv("MSL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MSL_OUTPUT_LANGUAGE", "en")
    monkeypatch.setenv("MSL_ASR_LANGUAGE", "en")
    monkeypatch.setenv("MSL_DEFAULT_PIPELINE", "pilot_no_llm")

    cfg = apply_env_overrides(_base_cfg())

    assert cfg.llm_endpoint == "http://gpu-box:11434/api/generate"
    assert cfg.llm_model == "mistral:7b"
    assert cfg.output_language == "en"
    assert cfg.asr_language == "en"
    assert cfg.default_pipeline == "pilot_no_llm"


def test_integer_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSL_MIN_SPEAKERS", "2")
    monkeypatch.setenv("MSL_MAX_SPEAKERS", "6")

    cfg = apply_env_overrides(_base_cfg())

    assert cfg.min_speakers == 2
    assert cfg.max_speakers == 6


def test_empty_string_env_var_does_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSL_LLM_MODEL", "")

    cfg = apply_env_overrides(_base_cfg())

    assert cfg.llm_model == "llama3.1:8b"
