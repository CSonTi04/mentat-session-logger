from __future__ import annotations

import argparse
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.diarization import DiarizationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext
from mentat_session_logger.transcription import StubAsrBackend, TranscriptionStage, WhisperXBackend


def _preferred_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="hu")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--min-speakers", type=int, default=4)
    parser.add_argument("--max-speakers", type=int, default=8)
    args = parser.parse_args()

    workspace = Path.cwd()
    env = EnvironmentResolver(workspace).resolve(args.env)
    env.asr_language = args.language
    env.min_speakers = args.min_speakers
    env.max_speakers = args.max_speakers

    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    source = Path(args.audio)
    prepared_target = artifacts.audio_file(f"{args.session}_16k.wav")
    if source.resolve() != prepared_target.resolve():
        prepared_target.write_bytes(source.read_bytes())

    try:
        backend = WhisperXBackend(device=_preferred_torch_device())
    except Exception:
        backend = StubAsrBackend()

    stage = TranscriptionStage(backend=backend, model_name=args.model, language=args.language)
    context = SessionContext(env=env, session_id=args.session)
    stage.run(context, artifacts)
    DiarizationStage().run(context, artifacts)
    print("Wrote whisperx_output.json, transcript_raw.txt, diarized_raw.md, diarization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
