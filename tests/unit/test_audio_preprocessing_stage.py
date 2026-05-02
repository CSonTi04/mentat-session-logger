from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.audio import AudioPreprocessingStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext


class FakeAudioRunner:
    def validate(self) -> None:
        return None

    def to_mono_16k(self, input_audio: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(input_audio.read_bytes())


def test_prepare_audio_stage_writes_expected_output(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    artifacts = ArtifactStore(env.root, "session_001")
    artifacts.ensure_session_dirs()
    raw = artifacts.input_file("session_001_raw.wav")
    raw.write_bytes(b"RIFFstub")

    stage = AudioPreprocessingStage(runner=FakeAudioRunner())
    result = stage.run(SessionContext(env=env, session_id="session_001"), artifacts)

    out = artifacts.audio_file("session_001_16k.wav")
    assert out.exists()
    assert result.output_artifacts[0].path == out
