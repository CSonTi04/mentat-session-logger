from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import ensure_dir
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult


@dataclass
class AudioCommandRunner:
    ffmpeg_bin: str = "ffmpeg"

    def validate(self) -> None:
        if shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError("ffmpeg is required but not found in PATH")

    def to_mono_16k(self, input_audio: Path, output_wav: Path) -> None:
        ensure_dir(output_wav.parent)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(input_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def normalize_loudness(self, input_wav: Path, output_wav: Path) -> None:
        ensure_dir(output_wav.parent)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(input_wav),
            "-af",
            "loudnorm",
            str(output_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


class AudioPreprocessingStage:
    name = "prepare_audio"

    def __init__(self, runner: AudioCommandRunner | None = None) -> None:
        self.runner = runner or AudioCommandRunner()

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        self.runner.validate()
        input_candidates = sorted((artifacts.session_root / "input").glob("*"))
        if not input_candidates:
            raise FileNotFoundError("No input audio found in session input directory")
        input_audio = input_candidates[0]
        output_wav = artifacts.audio_file(f"{context.session_id}_16k.wav")
        self.runner.to_mono_16k(input_audio=input_audio, output_wav=output_wav)
        return StageResult(
            input_artifacts=[ArtifactRef("input_audio", input_audio)],
            output_artifacts=[ArtifactRef("prepared_audio", output_wav)],
            metadata={"ffmpeg": getattr(self.runner, "ffmpeg_bin", "custom-runner")},
        )


class AudioNormalizationStage:
    name = "normalize_audio"

    def __init__(self, runner: AudioCommandRunner | None = None) -> None:
        self.runner = runner or AudioCommandRunner()

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        self.runner.validate()
        prepared_wav = artifacts.audio_file(f"{context.session_id}_16k.wav")
        if not prepared_wav.exists():
            raise FileNotFoundError(f"Prepared audio missing: {prepared_wav}")
        normalized = artifacts.audio_file(f"{context.session_id}_16k_norm.wav")
        self.runner.normalize_loudness(prepared_wav, normalized)
        return StageResult(
            input_artifacts=[ArtifactRef("prepared_audio", prepared_wav)],
            output_artifacts=[ArtifactRef("normalized_audio", normalized)],
        )
