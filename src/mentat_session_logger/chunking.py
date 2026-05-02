from __future__ import annotations

import re
from dataclasses import dataclass

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import write_text
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult

LINE_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)-(\d\d:\d\d:\d\d)\]\s+([^:]+):\s?(.*)$")


@dataclass(frozen=True)
class Chunk:
    start: str
    end: str
    lines: list[str]


class TranscriptChunkingStage:
    name = "chunk_transcript"

    def __init__(self, target_minutes: int = 10) -> None:
        self.target_minutes = target_minutes

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        source = artifacts.transcript_file("diarized_named_corrected.md")
        if not source.exists():
            source = artifacts.transcript_file("diarized_named.md")
        if not source.exists():
            raise FileNotFoundError("No named transcript for chunking")

        lines = source.read_text(encoding="utf-8").splitlines()
        chunks = self._chunk_lines(lines, self.target_minutes * 60)
        output_artifacts = []
        for idx, chunk in enumerate(chunks, start=1):
            name = f"chunk_{idx:03d}.md"
            path = artifacts.chunk_file(name)
            body = (
                f"# Chunk {idx:03d}\n\nstart: {chunk.start}\nend: {chunk.end}\n\n"
                + "\n".join(chunk.lines)
            )
            write_text(path, body + "\n")
            output_artifacts.append(ArtifactRef(f"chunk_{idx:03d}", path))

        return StageResult(
            input_artifacts=[ArtifactRef("transcript", source)],
            output_artifacts=output_artifacts,
        )

    def _chunk_lines(self, lines: list[str], target_seconds: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        chunk_start = "00:00:00"
        last_end = "00:00:00"

        for line in lines:
            match = LINE_RE.match(line)
            if not match:
                if current:
                    current.append(line)
                continue

            start, end, _, _ = match.groups()
            if not current:
                chunk_start = start
            if current and (_to_seconds(end) - _to_seconds(chunk_start) > target_seconds):
                chunks.append(Chunk(start=chunk_start, end=last_end, lines=current.copy()))
                current = []
                chunk_start = start
            current.append(line)
            last_end = end

        if current:
            chunks.append(Chunk(start=chunk_start, end=last_end, lines=current.copy()))
        return chunks


def _to_seconds(stamp: str) -> int:
    hh, mm, ss = stamp.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)
