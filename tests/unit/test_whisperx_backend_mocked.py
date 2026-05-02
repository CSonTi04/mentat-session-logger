from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mentat_session_logger.transcription import WhisperXBackend


class _FakeModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def transcribe(self, audio: Any) -> dict[str, Any]:
        _ = audio
        return copy.deepcopy(self._payload)


class _FakeDiarizationResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        assert orient == "records"
        return self._records


def _install_fake_whisperx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_payload: dict[str, Any],
    aligned_payload: dict[str, Any] | None = None,
    align_error: Exception | None = None,
    diarization_records: list[dict[str, Any]] | None = None,
    assigned_payload: dict[str, Any] | None = None,
    call_log: dict[str, Any] | None = None,
) -> None:
    whisperx_mod = types.ModuleType("whisperx")
    diarize_mod = types.ModuleType("whisperx.diarize")

    def _load_model(model_name: str, device: str, language: str) -> _FakeModel:
        if call_log is not None:
            call_log["load_model"] = {
                "model_name": model_name,
                "device": device,
                "language": language,
            }
        return _FakeModel(base_payload)

    def _load_audio(audio_path: str) -> str:
        if call_log is not None:
            call_log["load_audio"] = audio_path
        return "AUDIO_BUFFER"

    def _load_align_model(language_code: str, device: str) -> tuple[str, dict[str, str]]:
        if call_log is not None:
            call_log["load_align_model"] = {"language_code": language_code, "device": device}
        return ("ALIGN_MODEL", {"kind": "meta"})

    def _align(
        segments: list[dict[str, Any]],
        align_model: str,
        metadata: dict[str, str],
        audio: str,
        device: str,
        return_char_alignments: bool,
    ) -> dict[str, Any]:
        if call_log is not None:
            call_log["align"] = {
                "segments_count": len(segments),
                "align_model": align_model,
                "metadata_kind": metadata.get("kind"),
                "audio": audio,
                "device": device,
                "return_char_alignments": return_char_alignments,
            }
        if align_error is not None:
            raise align_error
        if aligned_payload is not None:
            return copy.deepcopy(aligned_payload)
        return {"segments": copy.deepcopy(segments)}

    class _FakeDiarizationPipeline:
        def __init__(self, use_auth_token: str, device: str) -> None:
            if call_log is not None:
                call_log["diarization_init"] = {
                    "use_auth_token": use_auth_token,
                    "device": device,
                }

        def __call__(
            self,
            audio: str,
            min_speakers: int,
            max_speakers: int,
        ) -> _FakeDiarizationResult:
            if call_log is not None:
                call_log["diarization_call"] = {
                    "audio": audio,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
            return _FakeDiarizationResult(diarization_records or [])

    def _assign_word_speakers(
        diarized: _FakeDiarizationResult,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if call_log is not None:
            call_log["assign_word_speakers_called"] = True
            call_log["assign_word_speakers_diarized"] = diarized.to_dict("records")
        if assigned_payload is not None:
            return copy.deepcopy(assigned_payload)
        return result

    whisperx_mod.load_model = _load_model  # type: ignore[attr-defined]
    whisperx_mod.load_audio = _load_audio  # type: ignore[attr-defined]
    whisperx_mod.load_align_model = _load_align_model  # type: ignore[attr-defined]
    whisperx_mod.align = _align  # type: ignore[attr-defined]
    diarize_mod.DiarizationPipeline = _FakeDiarizationPipeline  # type: ignore[attr-defined]
    diarize_mod.assign_word_speakers = _assign_word_speakers  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "whisperx", whisperx_mod)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize_mod)


def test_whisperx_backend_without_hf_token_uses_word_speaker_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: dict[str, Any] = {"assign_word_speakers_called": False}
    base_payload = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"speaker": "SPEAKER_09"}, {"speaker": "SPEAKER_09"}],
            }
        ],
    }
    _install_fake_whisperx(monkeypatch, base_payload=base_payload, call_log=call_log)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    payload = WhisperXBackend(device="cpu").transcribe(
        audio_path=Path("fake.wav"),
        language="en",
        model_name="small.en",
        min_speakers=2,
        max_speakers=4,
    )

    assert payload["segments"][0]["speaker"] == "SPEAKER_09"
    assert payload["diarization"] == []
    assert any("set HF_TOKEN" in warning for warning in payload["warnings"])
    assert call_log["assign_word_speakers_called"] is False


def test_whisperx_backend_with_hf_token_runs_diarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: dict[str, Any] = {"assign_word_speakers_called": False}
    base_payload = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
    }
    assigned_payload = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_03"}]
    }
    records = [{"speaker": "SPEAKER_03", "start": 0.0, "end": 1.0}]
    _install_fake_whisperx(
        monkeypatch,
        base_payload=base_payload,
        assigned_payload=assigned_payload,
        diarization_records=records,
        call_log=call_log,
    )
    monkeypatch.setenv("HF_TOKEN", "token-123")

    payload = WhisperXBackend(device="cpu").transcribe(
        audio_path=Path("fake.wav"),
        language="en",
        model_name="small.en",
        min_speakers=1,
        max_speakers=3,
    )

    assert payload["segments"][0]["speaker"] == "SPEAKER_03"
    assert payload["diarization"] == records
    assert payload["warnings"] == []
    assert call_log["diarization_init"]["use_auth_token"] == "token-123"
    assert call_log["assign_word_speakers_called"] is True


def test_whisperx_backend_alignment_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_payload = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"speaker": "SPEAKER_01"}],
            }
        ],
    }
    _install_fake_whisperx(
        monkeypatch,
        base_payload=base_payload,
        align_error=RuntimeError("alignment crashed"),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    payload = WhisperXBackend(device="cpu").transcribe(
        audio_path=Path("fake.wav"),
        language="en",
        model_name="small.en",
        min_speakers=2,
        max_speakers=4,
    )

    assert payload["segments"][0]["speaker"] == "SPEAKER_01"
    assert any("alignment skipped: alignment crashed" in warning for warning in payload["warnings"])
