from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_json, read_yaml, write_yaml
from mentat_session_logger.models import (
    ArtifactRef,
    SessionContext,
    SpeakerMatchSuggestion,
    StageResult,
)


class SpeakerEmbeddingBackend(Protocol):
    def embedding_from_audio(self, wav_path: Path) -> NDArray[np.float32]:
        ...


@dataclass
class SimpleEmbeddingBackend:
    dimension: int = 32

    def embedding_from_audio(self, wav_path: Path) -> NDArray[np.float32]:
        data = wav_path.read_bytes()
        if not data:
            return np.zeros(self.dimension, dtype=np.float32)
        arr = np.frombuffer(data[: self.dimension * 8], dtype=np.uint8)
        out = np.zeros(self.dimension, dtype=np.float32)
        for idx, value in enumerate(arr):
            out[idx % self.dimension] += float(value)
        norm = np.linalg.norm(out)
        return out if norm == 0 else out / norm


@dataclass
class VoiceprintService:
    backend: SpeakerEmbeddingBackend

    def enroll_environment(self, env_root: Path) -> dict[str, str]:
        voiceprints_dir = env_root / "voiceprints"
        profiles_dir = env_root / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        profile_map: dict[str, str] = {}
        for person_dir in sorted(voiceprints_dir.iterdir() if voiceprints_dir.exists() else []):
            if not person_dir.is_dir():
                continue
            vectors: list[NDArray[np.float32]] = []
            for wav in sorted(person_dir.glob("*.wav")):
                vectors.append(self.backend.embedding_from_audio(wav))
            if not vectors:
                continue
            avg = np.mean(np.stack(vectors), axis=0)
            norm = np.linalg.norm(avg)
            avg = avg if norm == 0 else avg / norm
            out = profiles_dir / f"{person_dir.name}.npy"
            np.save(out, avg)
            profile_map[person_dir.name] = f"profiles/{person_dir.name}.npy"

        write_yaml(env_root / "speaker_profiles.yml", {"profiles": profile_map})
        return profile_map


class SpeakerMatchingStage:
    name = "match_speakers"

    def __init__(
        self,
        backend: SpeakerEmbeddingBackend | None = None,
        unknown_threshold: float = 0.65,
    ) -> None:
        self.backend = backend or SimpleEmbeddingBackend()
        self.unknown_threshold = unknown_threshold

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        diarization_path = artifacts.raw_file("diarization.json")
        if not diarization_path.exists():
            raise FileNotFoundError("Missing diarization.json")
        diarization = read_json(diarization_path)
        segments = diarization.get("segments", [])

        profile_cfg = context.env.root / "speaker_profiles.yml"
        profiles_payload = read_yaml(profile_cfg) if profile_cfg.exists() else {"profiles": {}}
        profiles = profiles_payload.get("profiles", {})

        loaded_profiles: dict[str, NDArray[np.float32]] = {}
        for person, rel in profiles.items():
            path = context.env.root / str(rel)
            if path.exists():
                loaded_profiles[str(person)] = np.load(path)

        speakers: dict[str, list[str]] = {}
        for segment in segments:
            key = str(segment.get("speaker", "SPEAKER_??"))
            speakers.setdefault(key, []).append(str(segment.get("text", "")))

        suggestions: dict[str, SpeakerMatchSuggestion] = {}
        for speaker_id, texts in speakers.items():
            pseudo = (
                context.env.root
                / "sessions"
                / context.session_id
                / "raw"
                / f"{_safe_filename(speaker_id)}.txt"
            )
            pseudo.write_text("\n".join(texts), encoding="utf-8")
            speaker_embedding = self.backend.embedding_from_audio(pseudo)

            best_name = "unknown"
            best_score = -1.0
            for person, profile_vec in loaded_profiles.items():
                score = _cosine_similarity(speaker_embedding, profile_vec)
                if score > best_score:
                    best_name = person
                    best_score = score
            if best_score < self.unknown_threshold:
                best_name = "unknown"
            suggestions[speaker_id] = SpeakerMatchSuggestion(best_name, max(best_score, 0.0))

        out_path = artifacts.map_file("speaker_match_suggestions.yml")
        write_yaml(
            out_path,
            {
                "speaker_match_suggestions": {
                    spk: {"best_match": s.best_match, "confidence": round(float(s.confidence), 3)}
                    for spk, s in suggestions.items()
                }
            },
        )

        return StageResult(
            input_artifacts=[ArtifactRef("diarization", diarization_path)],
            output_artifacts=[ArtifactRef("speaker_match_suggestions", out_path)],
        )


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
