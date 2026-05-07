from __future__ import annotations

import os

from mentat_session_logger.models import EnvironmentConfig

_STR_OVERRIDES: list[tuple[str, str]] = [
    ("MSL_LLM_PROVIDER", "llm_provider"),
    ("MSL_LLM_ENDPOINT", "llm_endpoint"),
    ("MSL_LLM_MODEL", "llm_model"),
    ("MSL_OUTPUT_LANGUAGE", "output_language"),
    ("MSL_ASR_LANGUAGE", "asr_language"),
    ("MSL_DEFAULT_PIPELINE", "default_pipeline"),
]

_INT_OVERRIDES: list[tuple[str, str]] = [
    ("MSL_MIN_SPEAKERS", "min_speakers"),
    ("MSL_MAX_SPEAKERS", "max_speakers"),
]


def _parse_int_env(var: str) -> int | None:
    """Parse an integer environment variable, raising a clear error when malformed."""
    raw = os.getenv(var)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{var} must be a valid integer, got {raw!r}") from None


def apply_env_overrides(cfg: EnvironmentConfig) -> EnvironmentConfig:
    """Override :class:`EnvironmentConfig` fields from ``MSL_*`` environment variables.

    This implements twelve-factor config (Factor III): deploy-time settings can be
    supplied via the process environment without modifying YAML files.  Any variable
    that is not set (or is set to an empty string) leaves the corresponding field
    unchanged.

    .. note::
        ``MSL_FFMPEG_BIN`` is *not* handled here.  It is consumed directly by
        :class:`~mentat_session_logger.audio.AudioCommandRunner` at instantiation time.

    Supported variables
    -------------------
    ``MSL_LLM_PROVIDER``
        LLM backend identifier, e.g. ``ollama``.
    ``MSL_LLM_ENDPOINT``
        Full URL of the LLM API endpoint,
        e.g. ``http://localhost:11434/api/generate``.
    ``MSL_LLM_MODEL``
        Model name passed to the LLM backend, e.g. ``llama3.1:8b``.
    ``MSL_OUTPUT_LANGUAGE``
        ISO 639-1 language code for generated text output, e.g. ``hu``.
    ``MSL_ASR_LANGUAGE``
        ISO 639-1 language code for speech recognition, e.g. ``hu``.
    ``MSL_DEFAULT_PIPELINE``
        Pipeline name used when ``--pipeline`` is omitted from the ``run`` command.
    ``MSL_MIN_SPEAKERS``
        Minimum speaker count hint for diarization (integer).
    ``MSL_MAX_SPEAKERS``
        Maximum speaker count hint for diarization (integer).
    """
    for env_var, attr in _STR_OVERRIDES:
        val = os.getenv(env_var)
        if val:
            setattr(cfg, attr, val)
    for env_var, attr in _INT_OVERRIDES:
        int_val = _parse_int_env(env_var)
        if int_val is not None:
            setattr(cfg, attr, int_val)
    return cfg
