#!/usr/bin/env python
"""
demo_pipeline.py — zero-dependency end-to-end demo run.

Runs the full mentat-session-logger pipeline using only stub backends.
No ffmpeg, no WhisperX, no Ollama, no HF_TOKEN is needed.

Usage
-----
    python scripts/demo_pipeline.py                   # defaults
    python scripts/demo_pipeline.py --env demo --session session_demo
    python scripts/demo_pipeline.py --pipeline pilot_no_llm

What it does
------------
1. Initialises a private environment under envs/<env>/ (if absent).
2. Places a minimal stub audio file in the session input folder.
3. Runs the pipeline with:
   - FakeAudioRunner  - skips ffmpeg entirely
   - StubAsrBackend   - returns a synthetic two-speaker transcript
   - SimpleEmbeddingBackend - no speechbrain required
4. Prints the key output files so you can inspect them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the package is importable when the repo is not installed
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mentat_session_logger.artifacts import ArtifactStore  # noqa: E402
from mentat_session_logger.campaign_memory import MemoryUpdateProposalStage  # noqa: E402
from mentat_session_logger.chunking import TranscriptChunkingStage  # noqa: E402
from mentat_session_logger.classification import (  # noqa: E402
    ChunkSummarizationStage,
    FinalNotebookGenerationStage,
    TopicClassificationStage,
)
from mentat_session_logger.diarization import DiarizationStage  # noqa: E402
from mentat_session_logger.environments import EnvironmentResolver  # noqa: E402
from mentat_session_logger.models import SessionContext  # noqa: E402
from mentat_session_logger.pipeline import PipelineRunner, load_pipeline_config  # noqa: E402
from mentat_session_logger.prompts import PromptRenderer  # noqa: E402
from mentat_session_logger.transcript import (  # noqa: E402
    GlossaryCorrectionStage,
    SpeakerMapApplicationStage,
)
from mentat_session_logger.transcription import StubAsrBackend, TranscriptionStage  # noqa: E402
from mentat_session_logger.voiceprints import (  # noqa: E402
    SimpleEmbeddingBackend,
    SpeakerMatchingStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub audio runner — replaces AudioCommandRunner so ffmpeg is not needed
# ---------------------------------------------------------------------------


class _StubAudioRunner:
    """Copies the source bytes as-is; bypasses ffmpeg entirely."""

    ffmpeg_bin: str = "stub-ffmpeg"

    def validate(self) -> None:
        return None

    def to_mono_16k(self, input_audio: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(input_audio.read_bytes())

    def normalize_loudness(self, input_wav: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(input_wav.read_bytes())


# ---------------------------------------------------------------------------
# Stub LLM — returns minimal well-formed responses without Ollama
# ---------------------------------------------------------------------------
import json  # noqa: E402


class _StubLlm:
    """Returns plausible stub responses for classification and summarisation."""

    def generate(self, prompt: str) -> str:
        if "Classify this transcript chunk." in prompt:
            return json.dumps(
                {
                    "start": "00:00:00",
                    "end": "00:00:04",
                    "primary_category": "IC_GAMEPLAY",
                    "secondary_categories": [],
                    "campaign_relevant": True,
                    "include_in_campaign_notebook": True,
                    "include_in_table_diary": False,
                    "include_in_rules_meta": False,
                    "summary": "A short in-character exchange.",
                    "notable_quotes": ["Stub transcript."],
                    "canon_facts": ["The party proceeds without incident."],
                    "rules_notes": [],
                    "off_topic_reason": "",
                }
            )
        if "Summarize this classified transcript chunk" in prompt:
            return (
                "### Category\nIC_GAMEPLAY\n\n"
                "### Short Summary\nA short in-character exchange.\n\n"
                "### Canon Facts\n- The party proceeds without incident.\n\n"
                "### Character Actions\n- none\n\n"
                "### NPCs\n- none\n\n"
                "### Rules/Meta\n- none\n\n"
                "### Off-Topic Notes\n- none\n\n"
                "### Uncertainty Notes\n- none\n"
            )
        # Glossary correction: echo the transcript back unchanged
        marker = "\n\nTranscript:\n"
        if marker in prompt:
            return prompt.split(marker, maxsplit=1)[1].strip()
        return "ok"


# ---------------------------------------------------------------------------
# Stage registry for the demo run
# ---------------------------------------------------------------------------
from mentat_session_logger.audio import (  # noqa: E402
    AudioNormalizationStage,
    AudioPreprocessingStage,
)


def _build_stage_registry(pipeline_name: str) -> dict:
    """Build a stage registry wired with stub backends."""
    stub_llm = _StubLlm()
    prompts = PromptRenderer(_REPO_ROOT / "prompts")
    return {
        "prepare_audio": AudioPreprocessingStage(runner=_StubAudioRunner()),
        "normalize_audio": AudioNormalizationStage(runner=_StubAudioRunner()),
        "transcribe": TranscriptionStage(backend=StubAsrBackend()),
        "diarize": DiarizationStage(),
        "match_speakers": SpeakerMatchingStage(backend=SimpleEmbeddingBackend()),
        "apply_speaker_map": SpeakerMapApplicationStage(),
        "glossary_correct": GlossaryCorrectionStage(stub_llm, prompts, _REPO_ROOT),
        "chunk_transcript": TranscriptChunkingStage(target_minutes=1),
        "classify_chunks": TopicClassificationStage(stub_llm, prompts),
        "summarize_chunks": ChunkSummarizationStage(stub_llm, prompts),
        "generate_final_outputs": FinalNotebookGenerationStage(),
        "propose_memory_update": MemoryUpdateProposalStage(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Zero-dependency end-to-end pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env", default="demo", help="Environment name (default: demo)")
    parser.add_argument(
        "--session", default="session_demo", help="Session ID (default: session_demo)"
    )
    parser.add_argument(
        "--pipeline",
        default="pilot_no_llm",
        help="Pipeline config name (default: pilot_no_llm)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use the full 'default' pipeline with stub LLM responses instead of pilot_no_llm",
    )
    args = parser.parse_args()

    pipeline_name = "default" if args.with_llm else args.pipeline

    # ------------------------------------------------------------------
    # 1. Initialise the environment
    # ------------------------------------------------------------------
    resolver = EnvironmentResolver(_REPO_ROOT)
    env_root = _REPO_ROOT / "envs" / args.env
    if not env_root.exists():
        logger.info("Creating environment: %s", args.env)
        resolver.init_env(args.env)
    env = resolver.resolve(args.env)

    # ------------------------------------------------------------------
    # 2. Prepare session directories and a stub input audio file
    # ------------------------------------------------------------------
    context = SessionContext(env=env, session_id=args.session)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    stub_audio = artifacts.input_file(f"{args.session}_raw.wav")
    if not stub_audio.exists():
        # Minimal valid-looking WAV header — enough for the stub runner to copy.
        stub_audio.write_bytes(b"RIFFstub")
        logger.info("Wrote stub input audio: %s", stub_audio)

    # ------------------------------------------------------------------
    # 3. Run the pipeline
    # ------------------------------------------------------------------
    logger.info("Running pipeline '%s' with stub backends …", pipeline_name)
    registry = _build_stage_registry(pipeline_name)
    pipeline_cfg = load_pipeline_config(_REPO_ROOT, pipeline_name)
    PipelineRunner(registry).run(context, pipeline_cfg)

    # ------------------------------------------------------------------
    # 4. Print output summary
    # ------------------------------------------------------------------
    session_root = artifacts.session_root
    _print_separator()
    print(f"✅  Demo pipeline complete — outputs in:\n    {session_root}\n")

    _show_file("Session notebook", artifacts.final_file("session_notebook.md"), lines=20)
    _show_file("Canon delta", artifacts.final_file("canon_delta.md"), lines=10)
    _show_file("Memory proposal", artifacts.final_file("memory_update_proposal.yml"), lines=15)
    _print_separator()

    return 0


def _print_separator() -> None:
    print("-" * 72)


def _show_file(label: str, path: Path, lines: int = 10) -> None:
    if not path.exists():
        print(f"  {label}: (not generated)\n")
        return
    content = path.read_text(encoding="utf-8")
    shown = "\n".join(content.splitlines()[:lines])
    suffix = "…" if len(content.splitlines()) > lines else ""
    print(f"  ── {label} ({path.name}) ──")
    print(shown + suffix)
    print()


if __name__ == "__main__":
    raise SystemExit(main())
