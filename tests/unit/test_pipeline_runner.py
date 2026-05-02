from pathlib import Path

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.models import ArtifactRef, EnvironmentConfig, SessionContext, StageResult
from mentat_session_logger.pipeline import (
    PipelineConfig,
    PipelineError,
    PipelineRunner,
    PipelineStageSpec,
)


class FakeStage:
    def __init__(self, name: str, marker: Path) -> None:
        self.name = name
        self.marker = marker

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        self.marker.write_text(self.name, encoding="utf-8")
        out = artifacts.manifest_file(f"out_{self.name}")
        out.write_text("ok", encoding="utf-8")
        return StageResult([], [ArtifactRef(self.name, out)])


def _ctx(tmp_path: Path) -> SessionContext:
    env_root = tmp_path / "envs" / "local"
    (env_root / "sessions").mkdir(parents=True)
    env = EnvironmentConfig(
        name="local",
        root=env_root,
        campaign_context_dir=env_root / "campaign_context",
        voiceprints_dir=env_root / "voiceprints",
        sessions_dir=env_root / "sessions",
        profiles_dir=env_root / "profiles",
    )
    return SessionContext(env=env, session_id="session_001")


def test_pipeline_respects_order(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    marker = tmp_path / "order.txt"
    stage_a = FakeStage("a", marker)
    stage_b = FakeStage("b", marker)
    runner = PipelineRunner({"a": stage_a, "b": stage_b})
    cfg = PipelineConfig("default", [PipelineStageSpec("a"), PipelineStageSpec("b")])
    runner.run(ctx, cfg)
    assert marker.read_text(encoding="utf-8") == "b"


def test_pipeline_skips_disabled_stage(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    marker = tmp_path / "disabled.txt"
    runner = PipelineRunner({"a": FakeStage("a", marker)})
    cfg = PipelineConfig("default", [PipelineStageSpec("a", enabled=False)])
    runner.run(ctx, cfg)
    assert not marker.exists()


def test_pipeline_unknown_stage_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    runner = PipelineRunner({})
    cfg = PipelineConfig("default", [PipelineStageSpec("missing")])
    with pytest.raises(PipelineError):
        runner.run(ctx, cfg)
