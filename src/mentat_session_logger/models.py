from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: Path


@dataclass
class EnvironmentConfig:
    name: str
    root: Path
    campaign_context_dir: Path
    voiceprints_dir: Path
    sessions_dir: Path
    profiles_dir: Path
    default_pipeline: str = "default"
    output_language: str = "hu"
    asr_language: str = "hu"
    min_speakers: int = 4
    max_speakers: int = 8
    llm_provider: str = "ollama"
    llm_endpoint: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3.1:8b"


@dataclass(frozen=True)
class PipelineStageSpec:
    stage: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    stages: list[PipelineStageSpec]


@dataclass
class SessionContext:
    env: EnvironmentConfig
    session_id: str


@dataclass(frozen=True)
class SpeakerMatchSuggestion:
    best_match: str
    confidence: float


@dataclass(frozen=True)
class TranscriptSegment:
    start: str
    end: str
    speaker: str
    text: str


@dataclass(frozen=True)
class TopicClassification:
    start: str
    end: str
    primary_category: str
    secondary_categories: list[str]
    campaign_relevant: bool
    include_in_campaign_notebook: bool
    include_in_table_diary: bool
    include_in_rules_meta: bool
    summary: str
    notable_quotes: list[str]
    canon_facts: list[str]
    rules_notes: list[str]
    off_topic_reason: str


@dataclass
class MemoryUpdateProposal:
    session: str
    payload: dict[str, Any]


@dataclass
class StageManifest:
    stage_name: str
    started_at: str
    finished_at: str
    status: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    config: dict[str, Any]
    error_message: str | None = None


@dataclass(frozen=True)
class StageResult:
    input_artifacts: list[ArtifactRef]
    output_artifacts: list[ArtifactRef]
    metadata: dict[str, Any] = field(default_factory=dict)
