from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext
from mentat_session_logger.transcription import TranscriptionStage


class _FakeAsrBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_name: str,
        min_speakers: int,
        max_speakers: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "audio_path": audio_path,
                "language": language,
                "model_name": model_name,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        return {
            "segments": [
                {
                    "start": 1.2,
                    "end": 3.8,
                    "speaker": "SPEAKER_07",
                    "text": "Hello table.",
                }
            ]
        }


def _ctx_and_artifacts(tmp_path: Path) -> tuple[SessionContext, ArtifactStore]:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    context = SessionContext(env=env, session_id="session_001")
    artifacts = ArtifactStore(env.root, context.session_id)
    artifacts.ensure_session_dirs()
    return context, artifacts


def test_transcription_stage_prefers_normalized_audio_when_available(tmp_path: Path) -> None:
    context, artifacts = _ctx_and_artifacts(tmp_path)
    prepared = artifacts.audio_file("session_001_16k.wav")
    normalized = artifacts.audio_file("session_001_16k_norm.wav")
    prepared.write_bytes(b"prepared")
    normalized.write_bytes(b"normalized")

    backend = _FakeAsrBackend()
    stage = TranscriptionStage(backend=backend, model_name="small.en", language="en")
    result = stage.run(context, artifacts)

    assert backend.calls[0]["audio_path"] == normalized
    assert result.input_artifacts[0].path == normalized
    assert artifacts.raw_file("whisperx_output.json").exists()
    assert artifacts.raw_file("transcript_raw.txt").read_text(encoding="utf-8") == "Hello table.\n"
    assert (
        artifacts.transcript_file("diarized_raw.md").read_text(encoding="utf-8")
        == "[00:00:01-00:00:03] SPEAKER_07: Hello table.\n"
    )


def test_transcription_stage_raises_when_audio_missing(tmp_path: Path) -> None:
    context, artifacts = _ctx_and_artifacts(tmp_path)
    stage = TranscriptionStage(backend=_FakeAsrBackend())
    with pytest.raises(FileNotFoundError, match="Prepared audio missing"):
        stage.run(context, artifacts)
