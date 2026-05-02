from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import write_json, write_text
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult


class AsrBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_name: str,
        min_speakers: int,
        max_speakers: int,
    ) -> dict:
        ...


@dataclass
class WhisperXBackend:
    device: str = "cpu"

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_name: str,
        min_speakers: int,
        max_speakers: int,
    ) -> dict:
        try:
            import whisperx  # type: ignore
        except ImportError as exc:
            raise RuntimeError("whisperx is not installed; install optional runtime dependencies") from exc

        model = whisperx.load_model(model_name, self.device, language=language)
        result = model.transcribe(str(audio_path))

        # Alignment/diarization may fail depending on local setup; keep transcript usable.
        segments = result.get("segments", [])
        return {
            "language": language,
            "model": model_name,
            "segments": segments,
            "diarization": result.get("diarization", []),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }


@dataclass
class StubAsrBackend:
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_name: str,
        min_speakers: int,
        max_speakers: int,
    ) -> dict:
        return {
            "language": language,
            "model": model_name,
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "speaker": "SPEAKER_00",
                    "text": "Stub transcript. Replace with real WhisperX output.",
                }
            ],
            "diarization": [],
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "source_audio": str(audio_path),
        }


class TranscriptionStage:
    name = "transcribe"

    def __init__(self, backend: AsrBackend, model_name: str = "large-v3", language: str = "hu") -> None:
        self.backend = backend
        self.model_name = model_name
        self.language = language

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        prepared = artifacts.audio_file(f"{context.session_id}_16k.wav")
        normalized = artifacts.audio_file(f"{context.session_id}_16k_norm.wav")
        source = normalized if normalized.exists() else prepared
        if not source.exists():
            raise FileNotFoundError("Prepared audio missing; run prepare_audio first")

        payload = self.backend.transcribe(
            audio_path=source,
            language=context.env.asr_language or self.language,
            model_name=self.model_name,
            min_speakers=context.env.min_speakers,
            max_speakers=context.env.max_speakers,
        )
        raw_json = artifacts.raw_file("whisperx_output.json")
        write_json(raw_json, payload)

        raw_lines: list[str] = []
        diarized_lines: list[str] = []
        for segment in payload.get("segments", []):
            start = _ts(float(segment.get("start", 0.0)))
            end = _ts(float(segment.get("end", 0.0)))
            speaker = str(segment.get("speaker", "SPEAKER_??"))
            text = str(segment.get("text", "")).strip()
            raw_lines.append(text)
            diarized_lines.append(f"[{start}-{end}] {speaker}: {text}")

        raw_transcript = artifacts.raw_file("transcript_raw.txt")
        diarized_md = artifacts.transcript_file("diarized_raw.md")
        write_text(raw_transcript, "\n".join(raw_lines).strip() + "\n")
        write_text(diarized_md, "\n".join(diarized_lines).strip() + "\n")

        return StageResult(
            input_artifacts=[ArtifactRef("prepared_audio", source)],
            output_artifacts=[
                ArtifactRef("whisperx_output", raw_json),
                ArtifactRef("transcript_raw", raw_transcript),
                ArtifactRef("diarized_raw", diarized_md),
            ],
        )


def _ts(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
