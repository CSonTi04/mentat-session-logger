"""Unit tests for the mentat-session-logger CLI.

Each test verifies that a sub-command dispatches to the expected stage.  The
heavy stage implementations are replaced with lightweight fakes so no external
tools (ffmpeg, WhisperX, Ollama) are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext, StageResult


# ---------------------------------------------------------------------------
# Minimal stage stand-in
# ---------------------------------------------------------------------------
class _RecordingStage:
    """Captures calls to run() without doing real work."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[SessionContext, ArtifactStore]] = []

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        self.calls.append((context, artifacts))
        return StageResult(input_artifacts=[], output_artifacts=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_env(tmp_path: Path) -> tuple[Path, str, str]:
    """Return (workspace_root, env_name, session_id) after creating local env."""
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    return tmp_path, "local", "session_cli_test"


def _run_cli(argv: list[str], monkeypatch: pytest.MonkeyPatch, **stage_patches: Any) -> int:
    """
    Patch the given stage constructors, then invoke the CLI and return its exit code.
    ``stage_patches`` maps an attribute name inside ``mentat_session_logger.cli`` to a
    callable that returns a recording stage.
    """
    import mentat_session_logger.cli as cli_mod

    for attr, factory in stage_patches.items():
        monkeypatch.setattr(cli_mod, attr, factory)

    from mentat_session_logger.cli import main

    return main(argv)


# ---------------------------------------------------------------------------
# init-env
# ---------------------------------------------------------------------------
def test_init_env_creates_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from mentat_session_logger.cli import main

    rc = main(["init-env", "--name", "my_campaign"])
    assert rc == 0
    assert (tmp_path / "envs" / "my_campaign" / "config.yml").exists()


# ---------------------------------------------------------------------------
# prepare-audio
# ---------------------------------------------------------------------------
def test_prepare_audio_dispatches_to_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    stage = _RecordingStage("prepare_audio")
    monkeypatch.setattr(
        "mentat_session_logger.cli.AudioPreprocessingStage",
        lambda: stage,
    )
    rc = _run_cli(["prepare-audio", "--env", env, "--session", session], monkeypatch)
    assert rc == 0
    assert len(stage.calls) == 1
    assert stage.calls[0][0].session_id == session


# ---------------------------------------------------------------------------
# normalize-audio
# ---------------------------------------------------------------------------
def test_normalize_audio_dispatches_to_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    stage = _RecordingStage("normalize_audio")
    monkeypatch.setattr(
        "mentat_session_logger.cli.AudioNormalizationStage",
        lambda: stage,
    )
    rc = _run_cli(["normalize-audio", "--env", env, "--session", session], monkeypatch)
    assert rc == 0
    assert len(stage.calls) == 1
    assert stage.calls[0][0].session_id == session


# ---------------------------------------------------------------------------
# enroll-voiceprints
# ---------------------------------------------------------------------------
def test_enroll_voiceprints_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, env, _ = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    enrolled: list[str] = []

    class _FakeVoiceprintService:
        def enroll_environment(self, env_root: Path) -> dict[str, str]:
            enrolled.append(str(env_root))
            return {}

    monkeypatch.setattr(
        "mentat_session_logger.cli.VoiceprintService",
        lambda backend: _FakeVoiceprintService(),
    )
    from mentat_session_logger.cli import main

    rc = main(["enroll-voiceprints", "--env", env])
    assert rc == 0
    assert len(enrolled) == 1


# ---------------------------------------------------------------------------
# match-speakers
# ---------------------------------------------------------------------------
def test_match_speakers_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    stage = _RecordingStage("match_speakers")
    monkeypatch.setattr("mentat_session_logger.cli.SpeakerMatchingStage", lambda: stage)
    rc = _run_cli(["match-speakers", "--env", env, "--session", session], monkeypatch)
    assert rc == 0
    assert len(stage.calls) == 1


# ---------------------------------------------------------------------------
# apply-speaker-map
# ---------------------------------------------------------------------------
def test_apply_speaker_map_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    stage = _RecordingStage("apply_speaker_map")
    monkeypatch.setattr("mentat_session_logger.cli.SpeakerMapApplicationStage", lambda: stage)
    rc = _run_cli(["apply-speaker-map", "--env", env, "--session", session], monkeypatch)
    assert rc == 0
    assert len(stage.calls) == 1


# ---------------------------------------------------------------------------
# chunk-transcript
# ---------------------------------------------------------------------------
def test_chunk_transcript_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    stage = _RecordingStage("chunk_transcript")
    monkeypatch.setattr("mentat_session_logger.cli.TranscriptChunkingStage", lambda: stage)
    rc = _run_cli(["chunk-transcript", "--env", env, "--session", session], monkeypatch)
    assert rc == 0
    assert len(stage.calls) == 1


# ---------------------------------------------------------------------------
# generate-final-outputs
# ---------------------------------------------------------------------------
def test_generate_final_outputs_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, env, session = _init_env(tmp_path)
    monkeypatch.chdir(workspace)

    final_stage = _RecordingStage("generate_final_outputs")
    memory_stage = _RecordingStage("propose_memory_update")
    monkeypatch.setattr(
        "mentat_session_logger.cli.FinalNotebookGenerationStage", lambda: final_stage
    )
    monkeypatch.setattr(
        "mentat_session_logger.cli.MemoryUpdateProposalStage", lambda: memory_stage
    )
    rc = _run_cli(
        ["generate-final-outputs", "--env", env, "--session", session], monkeypatch
    )
    assert rc == 0
    assert len(final_stage.calls) == 1
    assert len(memory_stage.calls) == 1
