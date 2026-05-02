"""
Integration tests that run LLM stages against real AMI Corpus transcript data.

The tests use the AMI ES2008a words XML files that should already be present at:

    envs/local/sessions/session_ami_es2008a/raw/ami_public_manual_1.6.2/words/

All tests are skipped if those files are missing or Ollama is not running.

Run on laptop (phi3:mini):
    pytest tests/integration/test_llm_ami_data.py -m integration -v

Run on full rig (llama3.1:8b):
    pytest tests/integration/test_llm_ami_data.py -m integration --llm-profile rig -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.classification import ChunkSummarizationStage, TopicClassificationStage
from mentat_session_logger.chunking import TranscriptChunkingStage
from mentat_session_logger.io import write_text
from mentat_session_logger.llm import OllamaClient
from mentat_session_logger.models import SessionContext
from mentat_session_logger.prompts import PromptRenderer
from mentat_session_logger.transcript import GlossaryCorrectionStage

from tests.integration.ami_parser import build_diarized_transcript

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
AMI_WORDS_DIR = (
    REPO_ROOT
    / "envs/local/sessions/session_ami_es2008a/raw/ami_public_manual_1.6.2/words"
)
MEETING_ID = "ES2008a"

# How many seconds of the meeting to process per test run.
# Laptop: 5 min — fast enough for phi3:mini on CPU.
# Rig:   10 min — exercises more content for quality checks.
LAPTOP_SECONDS = 300.0
RIG_SECONDS = 600.0


# ---------------------------------------------------------------------------
# Module-level skip if AMI data absent
# ---------------------------------------------------------------------------
def _ami_available() -> bool:
    return (AMI_WORDS_DIR / f"{MEETING_ID}.A.words.xml").exists()


if not _ami_available():
    pytest.skip(
        "AMI ES2008a words XML not found. "
        f"Expected: {AMI_WORDS_DIR}\n"
        "Download from: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ami_max_seconds(llm_profile: str) -> float:
    return RIG_SECONDS if llm_profile == "rig" else LAPTOP_SECONDS


@pytest.fixture(scope="module")
def ami_transcript(ami_max_seconds: float) -> str:
    transcript = build_diarized_transcript(AMI_WORDS_DIR, MEETING_ID, ami_max_seconds)
    assert transcript, "AMI parser returned empty transcript"
    return transcript


@pytest.fixture()
def ami_artifacts(session_env, tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(session_env.root, "session_ami_es2008a")
    store.ensure_session_dirs()
    return store


@pytest.fixture()
def ami_context(session_env) -> SessionContext:
    return SessionContext(env=session_env, session_id="session_ami_es2008a")


# ---------------------------------------------------------------------------
# Test: parser produces valid diarized lines
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_ami_parser_output_format(ami_transcript: str) -> None:
    """Every line from the parser must match diarized transcript format."""
    import re

    LINE_RE = re.compile(r"^\[\d\d:\d\d:\d\d-\d\d:\d\d:\d\d\] Speaker_[A-D]: .+$")
    lines = [l for l in ami_transcript.splitlines() if l.strip()]
    assert len(lines) > 10, f"Too few transcript lines: {len(lines)}"
    bad = [l for l in lines if not LINE_RE.match(l)]
    assert not bad, f"Malformed lines:\n" + "\n".join(bad[:5])


# ---------------------------------------------------------------------------
# Test: GlossaryCorrectionStage on real AMI transcript
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_glossary_correction_on_ami(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    ami_transcript: str,
    ami_context: SessionContext,
    ami_artifacts: ArtifactStore,
) -> None:
    """Glossary correction stage runs end-to-end on real AMI transcript."""
    source = ami_artifacts.transcript_file("diarized_named.md")
    write_text(source, ami_transcript)

    stage = GlossaryCorrectionStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
        workspace_root=REPO_ROOT,
    )
    stage.run(ami_context, ami_artifacts)

    out = ami_artifacts.transcript_file("diarized_named_corrected.md")
    assert out.exists(), "Corrected transcript not written"
    corrected = out.read_text(encoding="utf-8")
    assert len(corrected.strip()) > 100, "Corrected transcript is suspiciously short"


# ---------------------------------------------------------------------------
# Test: full chunk→classify→summarize pipeline on real AMI data
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_classify_and_summarize_ami_chunks(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    ami_transcript: str,
    ami_context: SessionContext,
    ami_artifacts: ArtifactStore,
) -> None:
    """
    Chunk the real AMI transcript then classify and summarize each chunk.
    Validates that the LLM produces valid JSON for every chunk.
    """
    # Write transcript as the named (non-corrected) form so chunking picks it up
    named = ami_artifacts.transcript_file("diarized_named.md")
    write_text(named, ami_transcript)

    # Chunk into ~5-minute windows
    chunk_result = TranscriptChunkingStage(target_minutes=5).run(ami_context, ami_artifacts)
    chunk_files = sorted(ami_artifacts.chunks_dir().glob("chunk_*.md"))
    assert chunk_files, "No chunks produced from AMI transcript"

    # Classify all chunks
    TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(ami_context, ami_artifacts)

    # Summarize all chunks
    ChunkSummarizationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(ami_context, ami_artifacts)

    # Validate every classification file
    class_files = sorted((ami_artifacts.session_root / "classifications").glob("chunk_*.json"))
    assert class_files, "No classification files produced"

    valid_categories = {
        "IC_GAMEPLAY", "OOC_RULES", "OOC_STRATEGY", "LORE_OR_RECAP",
        "TABLE_SOCIAL", "LOGISTICS", "TECHNICAL_AUDIO", "AMBIGUOUS",
    }
    failures: list[str] = []
    for cf in class_files:
        try:
            payload = json.loads(cf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{cf.name}: invalid JSON — {exc}")
            continue
        cat = payload.get("primary_category", "MISSING")
        if cat not in valid_categories:
            failures.append(f"{cf.name}: unexpected category '{cat}'")

    assert not failures, "Classification failures:\n" + "\n".join(failures)

    # Validate every summary file
    summary_files = sorted(
        (ami_artifacts.session_root / "chunk_summaries").glob("chunk_*_summary.md")
    )
    assert summary_files, "No summary files produced"
    for sf in summary_files:
        content = sf.read_text(encoding="utf-8").strip()
        assert len(content) > 10, f"{sf.name} summary is empty"


# ---------------------------------------------------------------------------
# Rig-only: category distribution sanity check
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.rig
def test_ami_classification_distribution_on_rig(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    ami_transcript: str,
    ami_context: SessionContext,
    ami_artifacts: ArtifactStore,
    llm_profile: str,
) -> None:
    """
    On the rig model, the AMI meeting (a real project-planning meeting)
    should produce at least one TABLE_SOCIAL or LOGISTICS chunk within
    the first 10 minutes — it opens with introductions and admin.
    """
    if llm_profile != "rig":
        pytest.skip("Distribution check only runs with --llm-profile rig")

    named = ami_artifacts.transcript_file("diarized_named.md")
    write_text(named, ami_transcript)
    TranscriptChunkingStage(target_minutes=5).run(ami_context, ami_artifacts)
    TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(ami_context, ami_artifacts)

    categories = []
    for cf in sorted((ami_artifacts.session_root / "classifications").glob("chunk_*.json")):
        try:
            payload = json.loads(cf.read_text(encoding="utf-8"))
            categories.append(payload.get("primary_category", ""))
        except json.JSONDecodeError:
            pass

    social_or_logistics = {"TABLE_SOCIAL", "LOGISTICS", "OOC_STRATEGY", "LORE_OR_RECAP"}
    found = [c for c in categories if c in social_or_logistics]
    assert found, (
        f"Expected at least one social/logistics/strategy chunk in the opening of ES2008a. "
        f"Got categories: {categories}"
    )
