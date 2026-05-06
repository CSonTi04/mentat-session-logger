from __future__ import annotations

import json
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.audio import AudioNormalizationStage, AudioPreprocessingStage
from mentat_session_logger.campaign_memory import MemoryUpdateProposalStage
from mentat_session_logger.chunking import TranscriptChunkingStage
from mentat_session_logger.classification import (
    ChunkSummarizationStage,
    FinalNotebookGenerationStage,
    TopicClassificationStage,
)
from mentat_session_logger.diarization import DiarizationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.io import read_text, read_yaml
from mentat_session_logger.models import SessionContext
from mentat_session_logger.pipeline import PipelineRunner, load_pipeline_config
from mentat_session_logger.prompts import PromptRenderer
from mentat_session_logger.transcript import GlossaryCorrectionStage, SpeakerMapApplicationStage
from mentat_session_logger.transcription import StubAsrBackend, TranscriptionStage
from mentat_session_logger.voiceprints import SpeakerMatchingStage


class _FakeAudioRunner:
    ffmpeg_bin: str = "fake-ffmpeg"

    def validate(self) -> None:
        return None

    def to_mono_16k(self, input_audio: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(input_audio.read_bytes())

    def normalize_loudness(self, input_wav: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(input_wav.read_bytes())


class _FakeLlm:
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
                    "include_in_rules_meta": True,
                    "summary": "A quick in-character exchange.",
                    "notable_quotes": ["Stub transcript."],
                    "canon_facts": ["The party receives a sealed message."],
                    "rules_notes": ["The GM requested a perception check."],
                    "off_topic_reason": "",
                }
            )
        if "Summarize this classified transcript chunk for campaign operations." in prompt:
            return (
                "### Category\nIC_GAMEPLAY\n\n"
                "### Short Summary\nA quick in-character exchange.\n\n"
                "### Canon Facts\n- The party receives a sealed message.\n\n"
                "### Character Actions\n- SPEAKER_00 delivers a short update.\n\n"
                "### NPCs\n- none\n\n"
                "### Rules/Meta\n- The GM requested a perception check.\n\n"
                "### Off-Topic Notes\n- none\n\n"
                "### Uncertainty Notes\n- none\n"
            )
        marker = "\n\nTranscript:\n"
        if marker in prompt:
            return prompt.split(marker, maxsplit=1)[1].strip()
        return "ok"


def test_default_pipeline_runs_end_to_end_with_mocked_dependencies(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    context = SessionContext(env=env, session_id="session_e2e")
    artifacts = ArtifactStore(env.root, context.session_id)
    artifacts.ensure_session_dirs()

    raw_input = artifacts.input_file(f"{context.session_id}_raw.wav")
    raw_input.write_bytes(b"RIFFstub")

    repo_root = Path(__file__).resolve().parents[2]
    prompt_renderer = PromptRenderer(repo_root / "prompts")
    fake_llm = _FakeLlm()

    stage_registry = {
        "prepare_audio": AudioPreprocessingStage(runner=_FakeAudioRunner()),
        "normalize_audio": AudioNormalizationStage(runner=_FakeAudioRunner()),
        "transcribe": TranscriptionStage(
            backend=StubAsrBackend(),
            model_name="small.en",
            language="en",
        ),
        "diarize": DiarizationStage(),
        "match_speakers": SpeakerMatchingStage(),
        "apply_speaker_map": SpeakerMapApplicationStage(),
        "glossary_correct": GlossaryCorrectionStage(fake_llm, prompt_renderer, repo_root),
        "chunk_transcript": TranscriptChunkingStage(target_minutes=1),
        "classify_chunks": TopicClassificationStage(fake_llm, prompt_renderer),
        "summarize_chunks": ChunkSummarizationStage(fake_llm, prompt_renderer),
        "generate_final_outputs": FinalNotebookGenerationStage(),
        "propose_memory_update": MemoryUpdateProposalStage(),
    }

    pipeline = load_pipeline_config(repo_root, "default")
    PipelineRunner(stage_registry).run(context, pipeline)

    canon_delta = read_text(artifacts.final_file("canon_delta.md"))
    rules_meta = read_text(artifacts.final_file("rules_and_meta.md"))
    notebook = read_text(artifacts.final_file("session_notebook.md"))
    memory = read_yaml(artifacts.final_file("memory_update_proposal.yml"))

    assert "The party receives a sealed message." in canon_delta
    assert "The GM requested a perception check." in rules_meta
    assert "A quick in-character exchange." in notebook
    candidates = memory.get("new_canon_candidates", [])
    assert isinstance(candidates, list) and candidates
    assert candidates[0]["fact"] == "The party receives a sealed message."
