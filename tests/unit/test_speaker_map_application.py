from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.io import write_text, write_yaml
from mentat_session_logger.models import SessionContext
from mentat_session_logger.transcript import SpeakerMapApplicationStage


def test_speaker_map_rewrites_labels_and_preserves_timestamps(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    ctx = SessionContext(env=env, session_id="session_012")
    artifacts = ArtifactStore(env.root, "session_012")
    artifacts.ensure_session_dirs()

    write_text(
        artifacts.transcript_file("diarized_raw.md"),
        "[00:00:01-00:00:03] SPEAKER_00: Szia\n",
    )
    write_yaml(artifacts.map_file("speaker_map.yml"), {"speaker_map": {"SPEAKER_00": "GM"}})

    SpeakerMapApplicationStage().run(ctx, artifacts)
    out = artifacts.transcript_file("diarized_named.md").read_text(encoding="utf-8")
    assert "[00:00:01-00:00:03] GM: Szia" in out
