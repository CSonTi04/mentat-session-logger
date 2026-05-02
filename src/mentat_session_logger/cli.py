from __future__ import annotations

import argparse
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.audio import AudioNormalizationStage, AudioPreprocessingStage
from mentat_session_logger.campaign_memory import (
    ApprovedMemoryApplyStage,
    MemoryUpdateProposalStage,
)
from mentat_session_logger.chunking import TranscriptChunkingStage
from mentat_session_logger.classification import (
    ChunkSummarizationStage,
    FinalNotebookGenerationStage,
    TopicClassificationStage,
)
from mentat_session_logger.diarization import DiarizationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.llm import OllamaClient
from mentat_session_logger.models import SessionContext
from mentat_session_logger.pipeline import PipelineRunner, PipelineStage, load_pipeline_config
from mentat_session_logger.prompts import PromptRenderer
from mentat_session_logger.transcript import GlossaryCorrectionStage, SpeakerMapApplicationStage
from mentat_session_logger.transcription import StubAsrBackend, TranscriptionStage, WhisperXBackend
from mentat_session_logger.voiceprints import (
    SimpleEmbeddingBackend,
    SpeakerMatchingStage,
    VoiceprintService,
)


def _workspace_root() -> Path:
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mentat-session-logger")
    sub = parser.add_subparsers(dest="command", required=True)

    init_env = sub.add_parser("init-env", help="Initialize a private environment")
    init_env.add_argument("--name", required=True)
    init_env.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="Run configured pipeline")
    run.add_argument("--env", required=True)
    run.add_argument("--session", required=True)
    run.add_argument("--pipeline", default="default")
    run.add_argument("--resume", action="store_true")

    for cmd in [
        "prepare-audio",
        "transcribe",
        "enroll-voiceprints",
        "match-speakers",
        "apply-speaker-map",
        "glossary-correct",
        "chunk-transcript",
        "classify-chunks",
        "summarize-chunks",
        "generate-final-outputs",
        "apply-approved-memory-update",
    ]:
        p = sub.add_parser(cmd)
        p.add_argument("--env", required=True)
        if cmd != "enroll-voiceprints":
            p.add_argument("--session", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = _workspace_root()
    resolver = EnvironmentResolver(workspace)

    if args.command == "init-env":
        env_path = resolver.init_env(args.name, force=args.force)
        print(f"Initialized environment: {env_path}")
        return 0

    if args.command == "enroll-voiceprints":
        env = resolver.resolve(args.env)
        service = VoiceprintService(SimpleEmbeddingBackend())
        profiles = service.enroll_environment(env.root)
        print(f"Enrolled profiles: {sorted(profiles.keys())}")
        return 0

    env = resolver.resolve(args.env)
    context = SessionContext(env=env, session_id=args.session)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    llm = OllamaClient(endpoint=env.llm_endpoint, model=env.llm_model)
    prompts = PromptRenderer(workspace / "prompts")

    if args.command == "run":
        runner = PipelineRunner(_stage_registry(workspace, llm, prompts))
        pipeline = load_pipeline_config(workspace, args.pipeline)
        runner.run(context, pipeline, resume=args.resume)
        print("Pipeline complete")
        return 0

    if args.command == "prepare-audio":
        AudioPreprocessingStage().run(context, artifacts)
    elif args.command == "transcribe":
        _transcription_stage().run(context, artifacts)
        DiarizationStage().run(context, artifacts)
    elif args.command == "match-speakers":
        SpeakerMatchingStage().run(context, artifacts)
    elif args.command == "apply-speaker-map":
        SpeakerMapApplicationStage().run(context, artifacts)
    elif args.command == "glossary-correct":
        GlossaryCorrectionStage(llm, prompts, workspace).run(context, artifacts)
    elif args.command == "chunk-transcript":
        TranscriptChunkingStage().run(context, artifacts)
    elif args.command == "classify-chunks":
        TopicClassificationStage(llm, prompts).run(context, artifacts)
    elif args.command == "summarize-chunks":
        ChunkSummarizationStage(llm, prompts).run(context, artifacts)
    elif args.command == "generate-final-outputs":
        FinalNotebookGenerationStage().run(context, artifacts)
        MemoryUpdateProposalStage().run(context, artifacts)
    elif args.command == "apply-approved-memory-update":
        ApprovedMemoryApplyStage().run(context, artifacts)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(f"Command complete: {args.command}")
    return 0


def _transcription_stage() -> TranscriptionStage:
    try:
        backend = WhisperXBackend(device=_preferred_torch_device())
        return TranscriptionStage(backend=backend)
    except Exception:
        return TranscriptionStage(backend=StubAsrBackend())


def _preferred_torch_device() -> str:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stage_registry(
    workspace: Path,
    llm: OllamaClient,
    prompts: PromptRenderer,
) -> dict[str, PipelineStage]:
    return {
        "prepare_audio": AudioPreprocessingStage(),
        "normalize_audio": AudioNormalizationStage(),
        "transcribe": _transcription_stage(),
        "diarize": DiarizationStage(),
        "match_speakers": SpeakerMatchingStage(),
        "apply_speaker_map": SpeakerMapApplicationStage(),
        "glossary_correct": GlossaryCorrectionStage(llm, prompts, workspace),
        "chunk_transcript": TranscriptChunkingStage(),
        "classify_chunks": TopicClassificationStage(llm, prompts),
        "summarize_chunks": ChunkSummarizationStage(llm, prompts),
        "generate_final_outputs": FinalNotebookGenerationStage(),
        "propose_memory_update": MemoryUpdateProposalStage(),
        "apply_approved_memory_update": ApprovedMemoryApplyStage(),
    }
