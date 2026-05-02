from __future__ import annotations

from pathlib import Path

from mentat_session_logger.io import ensure_dir


class ArtifactStore:
    def __init__(self, env_root: Path, session_id: str) -> None:
        self.env_root = env_root
        self.session_id = session_id
        self.session_root = env_root / "sessions" / session_id

    def ensure_session_dirs(self) -> None:
        for folder in [
            "input",
            "audio",
            "raw",
            "maps",
            "transcripts",
            "chunks",
            "classifications",
            "chunk_summaries",
            "final",
            "manifests",
        ]:
            ensure_dir(self.session_root / folder)

    def input_file(self, filename: str) -> Path:
        return self.session_root / "input" / filename

    def audio_file(self, filename: str) -> Path:
        return self.session_root / "audio" / filename

    def raw_file(self, filename: str) -> Path:
        return self.session_root / "raw" / filename

    def map_file(self, filename: str) -> Path:
        return self.session_root / "maps" / filename

    def transcript_file(self, filename: str) -> Path:
        return self.session_root / "transcripts" / filename

    def chunks_dir(self) -> Path:
        return self.session_root / "chunks"

    def chunk_file(self, filename: str) -> Path:
        return self.session_root / "chunks" / filename

    def classification_file(self, filename: str) -> Path:
        return self.session_root / "classifications" / filename

    def chunk_summary_file(self, filename: str) -> Path:
        return self.session_root / "chunk_summaries" / filename

    def final_file(self, filename: str) -> Path:
        return self.session_root / "final" / filename

    def manifest_file(self, stage_name: str) -> Path:
        return self.session_root / "manifests" / f"{stage_name}.json"
