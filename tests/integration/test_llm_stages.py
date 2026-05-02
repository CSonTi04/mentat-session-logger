"""
Integration tests for all LLM-powered pipeline stages.

Run against the laptop profile (phi3:mini, default):
    pytest tests/integration/ -m integration -v

Run against the full-rig profile (llama3.1:8b):
    pytest tests/integration/ -m integration --llm-profile rig -v

All tests are skipped automatically when Ollama is not running
or the required model has not been pulled yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.classification import ChunkSummarizationStage, TopicClassificationStage
from mentat_session_logger.io import write_json, write_text
from mentat_session_logger.llm import OllamaClient
from mentat_session_logger.models import SessionContext
from mentat_session_logger.prompts import PromptRenderer
from mentat_session_logger.transcript import GlossaryCorrectionStage

# ---------------------------------------------------------------------------
# Shared sample content
# ---------------------------------------------------------------------------
SAMPLE_TRANSCRIPT_CHUNK = """\
[00:01:00-00:03:30] GM: The Harkonnen fleet drops out of foldspace above Arrakis.
[00:03:30-00:04:10] Player_1: I try to contact House Atreides command on the comms.
[00:04:10-00:04:45] Player_2: Can I make a Bene Gesserit Voice check here?
[00:04:45-00:05:00] GM: That is absolutely allowed. Roll it.
[00:05:00-00:05:20] Player_2: Got a seven. Is that enough?
[00:05:20-00:05:45] GM: More than enough. The guard hesitates, then steps aside.
"""

SAMPLE_CHUNK_CLASSIFICATION = {
    "start": "00:01:00",
    "end": "00:05:45",
    "primary_category": "IC_GAMEPLAY",
    "secondary_categories": ["OOC_RULES"],
    "campaign_relevant": True,
    "include_in_campaign_notebook": True,
    "include_in_table_diary": False,
    "include_in_rules_meta": False,
    "summary": "The Harkonnen fleet arrives; Player_2 uses Bene Gesserit Voice successfully.",
    "notable_quotes": ["The guard hesitates, then steps aside."],
    "canon_facts": ["Harkonnen fleet arrived above Arrakis"],
    "rules_notes": ["Bene Gesserit Voice check, result 7"],
    "off_topic_reason": "",
}

SAMPLE_DIARIZED_TRANSCRIPT = """\
[00:00:10-00:00:40] Player_A: Az Atreideszek zarándoklatot tesznek Arakinra.
[00:00:40-00:01:10] Player_B: Megpróbálok kapcsolatba lépni a Kwizatz Haderachhal.
[00:01:10-00:01:45] GM: A fregátok közelednek a déli pólus felé.
"""


# ---------------------------------------------------------------------------
# Smoke: raw LLM response
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_llm_smoke(ollama_client: OllamaClient, ollama_model: str) -> None:
    """Model returns a non-empty string for a trivial prompt."""
    reply = ollama_client.generate("Reply with exactly one word: ready")
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0, "LLM returned an empty response"


@pytest.mark.integration
def test_llm_profile_label(ollama_model: str, llm_profile: str) -> None:
    """Fixture wiring: confirm the profile maps to the expected model."""
    from tests.integration.conftest import PROFILES

    expected = PROFILES[llm_profile]["model"]
    assert ollama_model == expected


# ---------------------------------------------------------------------------
# Stage: GlossaryCorrectionStage
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_glossary_correction_returns_transcript(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
    tmp_path: Path,
) -> None:
    """Corrected transcript is written and contains timestamp markers."""
    workspace_root = Path(__file__).resolve().parents[2]

    # Write a named diarized transcript for the stage to consume
    source = session_artifacts.transcript_file("diarized_named.md")
    write_text(source, SAMPLE_DIARIZED_TRANSCRIPT)

    stage = GlossaryCorrectionStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
        workspace_root=workspace_root,
    )
    stage.run(session_context, session_artifacts)

    out_path = session_artifacts.transcript_file("diarized_named_corrected.md")
    assert out_path.exists(), "Corrected transcript file was not created"
    corrected = out_path.read_text(encoding="utf-8")
    # The output must still look like a diarized transcript
    assert "[" in corrected and "]" in corrected, (
        "Output does not look like a timestamped transcript"
    )


# ---------------------------------------------------------------------------
# Stage: TopicClassificationStage
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_topic_classification_produces_json(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
) -> None:
    """Each chunk produces a valid JSON classification file."""
    # Write one chunk file
    chunk_path = session_artifacts.chunks_dir() / "chunk_000.md"
    write_text(chunk_path, SAMPLE_TRANSCRIPT_CHUNK)

    stage = TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    )
    stage.run(session_context, session_artifacts)

    class_file = session_artifacts.classification_file("chunk_000.json")
    assert class_file.exists(), "Classification JSON was not created"
    payload = json.loads(class_file.read_text(encoding="utf-8"))
    assert "primary_category" in payload, "'primary_category' missing from classification"
    assert isinstance(payload.get("campaign_relevant"), bool), "'campaign_relevant' must be bool"


@pytest.mark.integration
def test_topic_classification_category_is_valid(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
) -> None:
    """The primary category returned by the LLM is one of the defined categories."""
    valid_categories = {
        "IC_GAMEPLAY",
        "OOC_RULES",
        "OOC_STRATEGY",
        "LORE_OR_RECAP",
        "TABLE_SOCIAL",
        "LOGISTICS",
        "TECHNICAL_AUDIO",
        "AMBIGUOUS",
    }

    chunk_path = session_artifacts.chunks_dir() / "chunk_000.md"
    write_text(chunk_path, SAMPLE_TRANSCRIPT_CHUNK)

    TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(session_context, session_artifacts)

    class_file = session_artifacts.classification_file("chunk_000.json")
    payload = json.loads(class_file.read_text(encoding="utf-8"))
    cat = payload.get("primary_category", "")
    assert cat in valid_categories, f"Unexpected category '{cat}'. Valid: {valid_categories}"


# ---------------------------------------------------------------------------
# Stage: ChunkSummarizationStage
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_chunk_summarization_produces_markdown(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
) -> None:
    """Summarization stage produces a non-empty Markdown file per chunk."""
    # Write classification input
    class_path = session_artifacts.classification_file("chunk_000.json")
    write_json(class_path, SAMPLE_CHUNK_CLASSIFICATION)

    stage = ChunkSummarizationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    )
    stage.run(session_context, session_artifacts)

    summary_file = session_artifacts.chunk_summary_file("chunk_000_summary.md")
    assert summary_file.exists(), "Summary Markdown file was not created"
    content = summary_file.read_text(encoding="utf-8").strip()
    assert len(content) > 20, f"Summary is too short to be meaningful: {content!r}"


# ---------------------------------------------------------------------------
# Profile-specific performance guards
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.laptop
def test_classify_completes_within_laptop_budget(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
    llm_profile: str,
) -> None:
    """Classification must complete within the laptop time budget (90 s per chunk)."""
    import time

    if llm_profile != "laptop":
        pytest.skip("Laptop-budget test only runs with --llm-profile laptop")

    chunk_path = session_artifacts.chunks_dir() / "chunk_000.md"
    write_text(chunk_path, SAMPLE_TRANSCRIPT_CHUNK)

    start = time.monotonic()
    TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(session_context, session_artifacts)
    elapsed = time.monotonic() - start

    assert elapsed < 90, f"Classification took {elapsed:.1f}s — too slow for laptop (budget: 90s)"


@pytest.mark.integration
@pytest.mark.rig
def test_classification_quality_on_rig(
    ollama_client: OllamaClient,
    prompt_renderer: PromptRenderer,
    session_context: SessionContext,
    session_artifacts: ArtifactStore,
    llm_profile: str,
) -> None:
    """
    On a full rig (llama3.1:8b), the Harkonnen attack chunk should be classified
    as IC_GAMEPLAY or LORE_OR_RECAP — not logistics or off-topic.
    """
    if llm_profile != "rig":
        pytest.skip("Quality assertion test only runs with --llm-profile rig")

    chunk_path = session_artifacts.chunks_dir() / "chunk_000.md"
    write_text(chunk_path, SAMPLE_TRANSCRIPT_CHUNK)

    TopicClassificationStage(
        llm_client=ollama_client,
        prompt_renderer=prompt_renderer,
    ).run(session_context, session_artifacts)

    class_file = session_artifacts.classification_file("chunk_000.json")
    payload = json.loads(class_file.read_text(encoding="utf-8"))
    cat = payload.get("primary_category", "")
    acceptable = {"IC_GAMEPLAY", "LORE_OR_RECAP", "OOC_RULES", "OOC_STRATEGY"}
    assert cat in acceptable, (
        f"Rig model misclassified Harkonnen combat as '{cat}'. "
        f"Expected one of {acceptable}."
    )
