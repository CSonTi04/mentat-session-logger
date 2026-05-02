from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        try:
            import whisperx  # type: ignore
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "whisperx is not installed; install optional runtime dependencies"
            ) from exc

        model = whisperx.load_model(model_name, self.device, language=language)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio)
        warnings: list[str] = []
        diarization_records: list[dict[str, Any]] = []

        # Try alignment first so diarization can be mapped to better timestamps/words.
        segments = cast(list[dict[str, Any]], result.get("segments", []))
        try:
            align_language = str(result.get("language", language))
            align_model, metadata = whisperx.load_align_model(
                language_code=align_language,
                device=self.device,
            )
            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            if isinstance(aligned, dict):
                result = aligned
                segments = cast(list[dict[str, Any]], result.get("segments", segments))
        except Exception as exc:
            warnings.append(f"alignment skipped: {exc}")

        # True speaker diarization requires a Hugging Face token.
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device,
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                if hasattr(diarize_segments, "to_dict"):
                    diarization_records = cast(
                        list[dict[str, Any]],
                        diarize_segments.to_dict("records"),
                    )
                result = assign_word_speakers(diarize_segments, result)
                segments = cast(list[dict[str, Any]], result.get("segments", segments))
            except Exception as exc:
                warnings.append(f"diarization skipped: {exc}")
        else:
            warnings.append("diarization skipped: set HF_TOKEN to enable speaker diarization")

        # Fill missing speaker labels from word-level speaker attribution when available.
        for segment in segments:
            if "speaker" not in segment or not segment.get("speaker"):
                segment["speaker"] = _speaker_from_words(segment)

        # Alignment/diarization may fail depending on local setup; keep transcript usable.
        return {
            "language": language,
            "model": model_name,
            "segments": segments,
            "diarization": diarization_records,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "warnings": warnings,
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
    ) -> dict[str, Any]:
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

    def __init__(
        self,
        backend: AsrBackend,
        model_name: str = "large-v3",
        language: str = "hu",
    ) -> None:
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
        segments = cast(list[dict[str, Any]], payload.get("segments", []))
        for segment in segments:
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


def _speaker_from_words(segment: dict[str, Any]) -> str:
    words = segment.get("words", [])
    if not isinstance(words, list):
        return "SPEAKER_??"
    counts: Counter[str] = Counter()
    for word in words:
        if not isinstance(word, dict):
            continue
        speaker = word.get("speaker")
        if isinstance(speaker, str) and speaker.strip():
            counts[speaker.strip()] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return "SPEAKER_??"
