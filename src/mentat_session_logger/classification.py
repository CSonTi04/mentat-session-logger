from __future__ import annotations

from typing import Any

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_json, read_text, write_json, write_text
from mentat_session_logger.llm import LlmClient, parse_json_response
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult
from mentat_session_logger.prompts import PromptRenderer

DEFAULT_CLASSIFICATION = {
    "start": "00:00:00",
    "end": "00:00:00",
    "primary_category": "AMBIGUOUS",
    "secondary_categories": [],
    "campaign_relevant": True,
    "include_in_campaign_notebook": True,
    "include_in_table_diary": False,
    "include_in_rules_meta": False,
    "summary": "",
    "notable_quotes": [],
    "canon_facts": [],
    "rules_notes": [],
    "off_topic_reason": "",
}


class TopicClassificationStage:
    name = "classify_chunks"

    def __init__(self, llm_client: LlmClient, prompt_renderer: PromptRenderer) -> None:
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        chunk_files = sorted(artifacts.chunks_dir().glob("chunk_*.md"))
        if not chunk_files:
            raise FileNotFoundError("No chunk files found")

        outputs: list[ArtifactRef] = []
        for chunk in chunk_files:
            content = read_text(chunk)
            prompt = self.prompt_renderer.render("topic_classifier_prompt.txt", {"CHUNK": content})
            raw_output = ""
            try:
                raw_output = self.llm_client.generate(prompt + "\n\nChunk:\n" + content)
                parsed = parse_json_response(raw_output)
                payload = {**DEFAULT_CLASSIFICATION, **parsed}
            except Exception:
                payload = DEFAULT_CLASSIFICATION.copy()
                failed = artifacts.classification_file(f"{chunk.stem}_failed_llm_output.txt")
                write_text(failed, raw_output)

            out = artifacts.classification_file(f"{chunk.stem}.json")
            write_json(out, payload)
            outputs.append(ArtifactRef(out.stem, out))

        return StageResult(
            input_artifacts=[ArtifactRef("chunks", artifacts.chunks_dir())],
            output_artifacts=outputs,
        )


class ChunkSummarizationStage:
    name = "summarize_chunks"

    def __init__(self, llm_client: LlmClient, prompt_renderer: PromptRenderer) -> None:
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        class_files = sorted((artifacts.session_root / "classifications").glob("chunk_*.json"))
        outputs: list[ArtifactRef] = []
        for class_file in class_files:
            payload = read_json(class_file)
            prompt = self.prompt_renderer.render("chunk_summary_prompt.txt", {"JSON": str(payload)})
            try:
                summary = self.llm_client.generate(prompt)
            except Exception:
                summary = self._fallback_summary(payload)

            out = artifacts.chunk_summary_file(f"{class_file.stem}_summary.md")
            write_text(out, summary.strip() + "\n")
            outputs.append(ArtifactRef(out.stem, out))

        return StageResult(
            input_artifacts=[
                ArtifactRef("classifications", artifacts.session_root / "classifications")
            ],
            output_artifacts=outputs,
        )

    @staticmethod
    def _fallback_summary(payload: dict[str, Any]) -> str:
        return (
            f"# Chunk Summary\n\n"
            f"- category: {payload.get('primary_category', 'AMBIGUOUS')}\n"
            f"- short summary: {payload.get('summary', '')}\n"
            f"- canon facts: {payload.get('canon_facts', [])}\n"
            f"- rules/meta: {payload.get('rules_notes', [])}\n"
            f"- off-topic notes: {payload.get('off_topic_reason', '')}\n"
        )


class FinalNotebookGenerationStage:
    name = "generate_final_outputs"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        summaries = sorted((artifacts.session_root / "chunk_summaries").glob("chunk_*_summary.md"))
        transcript_path = artifacts.transcript_file("diarized_named_corrected.md")
        if not transcript_path.exists():
            transcript_path = artifacts.transcript_file("diarized_named.md")

        summary_text = "\n\n".join(read_text(path) for path in summaries)
        full_transcript = read_text(transcript_path) if transcript_path.exists() else ""

        notebook = artifacts.final_file("session_notebook.md")
        rules_meta = artifacts.final_file("rules_and_meta.md")
        diary = artifacts.final_file("table_diary.md")
        full = artifacts.final_file("full_transcript.md")
        canon = artifacts.final_file("canon_delta.md")

        write_text(
            notebook,
            (
                f"# Session Notebook ({context.session_id})\n\n"
                f"## TL;DR\n\nGenerated from chunk summaries.\n\n{summary_text}\n"
            ),
        )
        write_text(
            rules_meta,
            (
                "# Rules and Meta\n\n"
                "Extract from chunk classifications where include_in_rules_meta=true.\n"
            ),
        )
        write_text(diary, "# Table Diary\n\nOff-topic/social/logistics/technical extracts.\n")
        write_text(full, full_transcript)
        write_text(
            canon,
            "# Canon Delta\n\n- new facts\n- updated facts\n- possible contradictions\n",
        )

        return StageResult(
            input_artifacts=[
                ArtifactRef("chunk_summaries", artifacts.session_root / "chunk_summaries")
            ],
            output_artifacts=[
                ArtifactRef("session_notebook", notebook),
                ArtifactRef("rules_and_meta", rules_meta),
                ArtifactRef("table_diary", diary),
                ArtifactRef("full_transcript", full),
                ArtifactRef("canon_delta", canon),
            ],
        )
