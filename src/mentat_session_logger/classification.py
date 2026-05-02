from __future__ import annotations

import re
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
RULES_CATEGORIES = {"OOC_RULES", "OOC_STRATEGY"}
DIARY_CATEGORIES = {"TABLE_SOCIAL", "LOGISTICS", "TECHNICAL_AUDIO"}
PLACEHOLDER_FACTS = {"new facts", "updated facts", "possible contradictions"}
PLACEHOLDER_LINES = {
    "none",
    "none extracted",
    "none mentioned",
    "none mentioned in this chunk",
    "none mentioned in this transcript chunk",
    "none applicable",
    "none significant",
    "n a",
    "na",
    "not applicable",
}
PLACEHOLDER_PREFIXES = (
    "none ",
    "none explicitly",
    "none noted",
    "not mentioned",
)
SUMMARY_SECTION_MAP = {
    "category": "category",
    "short summary": "short_summary",
    "canon facts": "canon_facts",
    "character actions": "character_actions",
    "npcs": "npcs",
    "rules meta": "rules_meta",
    "off topic notes": "off_topic_notes",
    "uncertainty notes": "uncertainty_notes",
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
                raw_output = self.llm_client.generate(prompt)
                parsed = parse_json_response(raw_output)
                payload = {**DEFAULT_CLASSIFICATION, **parsed}
                failed = artifacts.classification_file(f"{chunk.stem}_failed_llm_output.txt")
                if failed.exists():
                    failed.unlink()
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
            chunk_path = artifacts.chunk_file(f"{class_file.stem}.md")
            chunk_text = read_text(chunk_path) if chunk_path.exists() else ""
            prompt = self.prompt_renderer.render(
                "chunk_summary_prompt.txt",
                {
                    "JSON": str(payload),
                    "CHUNK": chunk_text,
                },
            )
            try:
                summary = self.llm_client.generate(prompt)
                summary = self._clean_summary_output(summary)
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

    @staticmethod
    def _clean_summary_output(summary: str) -> str:
        lines = summary.strip().splitlines()
        while lines and lines[0].strip().lower().startswith(
            (
                "here is",
                "here's",
                "i can",
                "sure",
            )
        ):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        cleaned = "\n".join(lines).strip()
        return cleaned if cleaned else summary.strip()


class FinalNotebookGenerationStage:
    name = "generate_final_outputs"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        summaries = sorted((artifacts.session_root / "chunk_summaries").glob("chunk_*_summary.md"))
        class_files = sorted((artifacts.session_root / "classifications").glob("chunk_*.json"))
        transcript_path = artifacts.transcript_file("diarized_named_corrected.md")
        if not transcript_path.exists():
            transcript_path = artifacts.transcript_file("diarized_named.md")

        summary_by_chunk = {
            path.stem.replace("_summary", ""): _parse_summary_sections(read_text(path))
            for path in summaries
        }
        summary_text = "\n\n".join(read_text(path) for path in summaries)
        full_transcript = read_text(transcript_path) if transcript_path.exists() else ""
        classifications = [(path.stem, read_json(path)) for path in class_files]

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
        write_text(rules_meta, _build_rules_and_meta(classifications, summary_by_chunk))
        write_text(diary, _build_table_diary(classifications, summary_by_chunk))
        write_text(full, full_transcript)
        write_text(canon, _build_canon_delta(classifications, summary_by_chunk))

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


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _build_rules_and_meta(
    classifications: list[tuple[str, dict[str, Any]]],
    summary_by_chunk: dict[str, dict[str, list[str]]],
) -> str:
    rules_notes: list[str] = []
    for chunk_id, payload in classifications:
        category = str(payload.get("primary_category", ""))
        include = bool(payload.get("include_in_rules_meta", False))
        if not (include or category in RULES_CATEGORIES):
            continue
        notes = _coerce_str_list(payload.get("rules_notes"))
        if not notes:
            notes = summary_by_chunk.get(chunk_id, {}).get("rules_meta", [])
        rules_notes.extend(_non_placeholder_lines(notes))

    deduped = list(dict.fromkeys(rules_notes))
    body = "\n".join(f"- {note}" for note in deduped) if deduped else "- none extracted"
    return f"# Rules and Meta\n\n{body}\n"


def _build_table_diary(
    classifications: list[tuple[str, dict[str, Any]]],
    summary_by_chunk: dict[str, dict[str, list[str]]],
) -> str:
    lines: list[str] = []
    for chunk_id, payload in classifications:
        category = str(payload.get("primary_category", ""))
        include = bool(payload.get("include_in_table_diary", False))
        if include or category in DIARY_CATEGORIES:
            start = str(payload.get("start", "00:00:00"))
            end = str(payload.get("end", "00:00:00"))
            summary = str(payload.get("summary", "")).strip()
            if not summary:
                fallback = summary_by_chunk.get(chunk_id, {}).get("short_summary", [])
                summary = fallback[0] if fallback else "no summary"
            lines.append(f"- [{start}-{end}] {category}: {summary}")

    body = "\n".join(lines) if lines else "- none extracted"
    return f"# Table Diary\n\n{body}\n"


def _build_canon_delta(
    classifications: list[tuple[str, dict[str, Any]]],
    summary_by_chunk: dict[str, dict[str, list[str]]],
) -> str:
    facts: list[str] = []
    for chunk_id, payload in classifications:
        chunk_facts = _coerce_str_list(payload.get("canon_facts"))
        if not chunk_facts:
            chunk_facts = summary_by_chunk.get(chunk_id, {}).get("canon_facts", [])
        for fact in _non_placeholder_lines(chunk_facts):
            normalized = re.sub(r"\s+", " ", fact).strip()
            if normalized and normalized.lower() not in PLACEHOLDER_FACTS:
                facts.append(normalized)

    deduped = list(dict.fromkeys(facts))
    if deduped:
        body = "\n".join(f"- {fact}" for fact in deduped)
    else:
        body = "- none extracted"
    return f"# Canon Delta\n\n{body}\n"


def _parse_summary_sections(summary: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading = _extract_heading_key(line)
        if heading is not None:
            current_key = heading
            sections.setdefault(current_key, [])
            continue

        inline_key, inline_value = _extract_inline_section(line)
        if inline_key is not None:
            current_key = inline_key
            sections.setdefault(current_key, [])
            if inline_value:
                sections[current_key].append(inline_value)
            continue

        list_item = _extract_list_item(line)
        if current_key is not None:
            sections[current_key].append(list_item if list_item else line)

    return {key: _non_placeholder_lines(values) for key, values in sections.items()}


def _extract_heading_key(line: str) -> str | None:
    match = re.match(r"^#{1,6}\s+(.+?)\s*:?\s*$", line)
    if match is None:
        match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
    if match is None:
        return None
    normalized = _normalize_label(match.group(1))
    return SUMMARY_SECTION_MAP.get(normalized)


def _extract_inline_section(line: str) -> tuple[str | None, str]:
    match = re.match(r"^\*\*(.+?)\*\*\s*:\s*(.*)$", line)
    if match is None:
        return None, ""
    normalized = _normalize_label(match.group(1))
    key = SUMMARY_SECTION_MAP.get(normalized)
    return key, match.group(2).strip() if key else ""


def _extract_list_item(line: str) -> str:
    match = re.match(r"^[-*]\s+(.*\S)\s*$", line)
    if match:
        return match.group(1).strip()
    return ""


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _non_placeholder_lines(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate:
            continue
        normalized = _normalize_label(candidate)
        if normalized in PLACEHOLDER_LINES or normalized.startswith(PLACEHOLDER_PREFIXES):
            continue
        cleaned.append(candidate)
    return cleaned
