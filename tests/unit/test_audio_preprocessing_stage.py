from pathlib import Path

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.audio import AudioPreprocessingStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext


class FakeAudioRunner:
    ffmpeg_bin: str = "fake-ffmpeg"

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


def test_audio_command_runner_validate_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mentat_session_logger.audio import AudioCommandRunner

    monkeypatch.delenv("MSL_FFMPEG_BIN", raising=False)
    runner = AudioCommandRunner(ffmpeg_bin="no-such-binary-xyz")
    with pytest.raises(FileNotFoundError, match="no-such-binary-xyz"):
        runner.validate()


def test_audio_command_runner_validate_picks_up_msl_ffmpeg_bin_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    from mentat_session_logger.audio import AudioCommandRunner

    real_bin = shutil.which("echo") or shutil.which("true")
    if real_bin is None:
        pytest.skip("No real binary available for test")

    monkeypatch.setenv("MSL_FFMPEG_BIN", real_bin)
    runner = AudioCommandRunner()
    assert runner.ffmpeg_bin == real_bin
    runner.validate()  # should not raise
