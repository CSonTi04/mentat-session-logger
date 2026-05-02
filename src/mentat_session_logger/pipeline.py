from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_yaml, write_json
from mentat_session_logger.models import (
    PipelineConfig,
    PipelineStageSpec,
    SessionContext,
    StageManifest,
    StageResult,
    utc_now_iso,
)


class PipelineStage(Protocol):
    name: str

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        ...


class PipelineError(RuntimeError):
    pass


class ManifestWriter:
    def write_success(
        self,
        artifacts: ArtifactStore,
        stage_name: str,
        started_at: str,
        stage_result: StageResult,
        config: dict[str, object],
    ) -> None:
        manifest = StageManifest(
            stage_name=stage_name,
            started_at=started_at,
            finished_at=utc_now_iso(),
            status="success",
            input_artifacts=[str(a.path) for a in stage_result.input_artifacts],
            output_artifacts=[str(a.path) for a in stage_result.output_artifacts],
            config={**config, "config_hash": self.config_hash(config)},
            error_message=None,
        )
        write_json(artifacts.manifest_file(stage_name), asdict(manifest))

    def write_failure(
        self,
        artifacts: ArtifactStore,
        stage_name: str,
        started_at: str,
        config: dict[str, object],
        error_message: str,
    ) -> None:
        manifest = StageManifest(
            stage_name=stage_name,
            started_at=started_at,
            finished_at=utc_now_iso(),
            status="failed",
            input_artifacts=[],
            output_artifacts=[],
            config={**config, "config_hash": self.config_hash(config)},
            error_message=error_message,
        )
        write_json(artifacts.manifest_file(stage_name), asdict(manifest))

    @staticmethod
    def config_hash(config: dict[str, object]) -> str:
        raw = repr(sorted(config.items())).encode("utf-8")
        return sha256(raw).hexdigest()


class PipelineRunner:
    def __init__(
        self,
        stages: dict[str, PipelineStage],
        manifest_writer: ManifestWriter | None = None,
    ) -> None:
        self.stages = stages
        self.manifest_writer = manifest_writer or ManifestWriter()

    def run(
        self,
        context: SessionContext,
        pipeline_config: PipelineConfig,
        resume: bool = False,
    ) -> None:
        artifacts = ArtifactStore(context.env.root, context.session_id)
        artifacts.ensure_session_dirs()

        for spec in pipeline_config.stages:
            if not spec.enabled:
                continue
            if spec.stage not in self.stages:
                raise PipelineError(f"Unknown stage: {spec.stage}")

            stage = self.stages[spec.stage]
            manifest_path = artifacts.manifest_file(stage.name)
            if resume and self._is_manifest_success(manifest_path):
                continue

            started_at = utc_now_iso()
            try:
                result = stage.run(context, artifacts)
                self.manifest_writer.write_success(
                    artifacts=artifacts,
                    stage_name=stage.name,
                    started_at=started_at,
                    stage_result=result,
                    config=spec.config,
                )
            except Exception as exc:
                self.manifest_writer.write_failure(
                    artifacts=artifacts,
                    stage_name=stage.name,
                    started_at=started_at,
                    config=spec.config,
                    error_message=str(exc),
                )
                raise PipelineError(f"Stage failed: {stage.name}: {exc}") from exc

    @staticmethod
    def _is_manifest_success(manifest_path: Path) -> bool:
        if not manifest_path.exists():
            return False
        manifest = read_yaml(manifest_path) if manifest_path.suffix in {".yml", ".yaml"} else None
        if manifest is not None:
            return manifest.get("status") == "success"

        data = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        return data.get("status") == "success"


def load_pipeline_config(workspace_root: Path, pipeline_name: str) -> PipelineConfig:
    path = workspace_root / "configs" / "pipelines" / f"{pipeline_name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    payload = read_yaml(path)
    raw_stages = payload.get("pipeline", [])
    if not isinstance(raw_stages, list):
        raise ValueError("pipeline must be a list")

    stages: list[PipelineStageSpec] = []
    for item in raw_stages:
        if not isinstance(item, dict):
            raise ValueError("each pipeline item must be an object")
        stages.append(
            PipelineStageSpec(
                stage=str(item.get("stage")),
                enabled=bool(item.get("enabled", True)),
                config=dict(item.get("config", {})),
            )
        )

    return PipelineConfig(name=pipeline_name, stages=stages)
