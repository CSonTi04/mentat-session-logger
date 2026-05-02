from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.chunking import TranscriptChunkingStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.io import write_text
from mentat_session_logger.models import SessionContext


def test_chunking_keeps_lines_intact(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    artifacts = ArtifactStore(env.root, "session_001")
    artifacts.ensure_session_dirs()
    write_text(
        artifacts.transcript_file("diarized_named.md"),
        "\n".join(
            [
                "[00:00:01-00:00:05] GM: A",
                "[00:09:00-00:09:30] Player: B",
                "[00:11:00-00:11:30] GM: C",
            ]
        ),
    )

    stage = TranscriptChunkingStage(target_minutes=10)
    stage.run(SessionContext(env=env, session_id="session_001"), artifacts)

    files = sorted(artifacts.chunks_dir().glob("chunk_*.md"))
    assert len(files) >= 2
    first = files[0].read_text(encoding="utf-8")
    assert "[00:00:01-00:00:05] GM: A" in first
