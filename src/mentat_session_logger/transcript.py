from __future__ import annotations

import re
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_text, read_yaml, write_text
from mentat_session_logger.llm import LlmClient
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult
from mentat_session_logger.prompts import PromptRenderer

LINE_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)-(\d\d:\d\d:\d\d)\]\s+([^:]+):\s?(.*)$")


class SpeakerMapApplicationStage:
    name = "apply_speaker_map"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        source = artifacts.transcript_file("diarized_raw.md")
        if not source.exists():
            raise FileNotFoundError("Missing diarized_raw.md")
        mapping_path = artifacts.map_file("speaker_map.yml")
        speaker_map = {}
        if mapping_path.exists():
            payload = read_yaml(mapping_path)
            speaker_map = dict(payload.get("speaker_map", {}))

        out_lines: list[str] = []
        for line in read_text(source).splitlines():
            match = LINE_RE.match(line)
            if not match:
                out_lines.append(line)
                continue
            start, end, speaker, text = match.groups()
            resolved = str(speaker_map.get(speaker, speaker))
            out_lines.append(f"[{start}-{end}] {resolved}: {text}")

        out_path = artifacts.transcript_file("diarized_named.md")
        write_text(out_path, "\n".join(out_lines).strip() + "\n")
        return StageResult(
            input_artifacts=[
                ArtifactRef("diarized_raw", source),
                ArtifactRef("speaker_map", mapping_path),
            ],
            output_artifacts=[ArtifactRef("diarized_named", out_path)],
        )


class GlossaryCorrectionStage:
    name = "glossary_correct"

    def __init__(
        self,
        llm_client: LlmClient,
        prompt_renderer: PromptRenderer,
        workspace_root: Path,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer
        self.workspace_root = workspace_root

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        source = artifacts.transcript_file("diarized_named.md")
        if not source.exists():
            raise FileNotFoundError("Missing diarized_named.md")

        transcript = read_text(source)
        glossary_path = context.env.campaign_context_dir / "glossary.yml"
        fallback_glossary = self.workspace_root / "configs" / "glossary.yml"
        glossary = (
            read_text(glossary_path)
            if glossary_path.exists()
            else read_text(fallback_glossary)
        )

        prompt = self.prompt_renderer.render(
            "glossary_correction_prompt.txt",
            {
                "TRANSCRIPT": transcript,
                "GLOSSARY": glossary,
            },
        )
        try:
            corrected = self.llm_client.generate(
                prompt + "\n\nGlossary:\n" + glossary + "\n\nTranscript:\n" + transcript
            )
            if "[" not in corrected or "]" not in corrected:
                corrected = transcript
        except Exception:
            corrected = transcript

        out_path = artifacts.transcript_file("diarized_named_corrected.md")
        write_text(out_path, corrected.strip() + "\n")
        return StageResult(
            input_artifacts=[ArtifactRef("diarized_named", source)],
            output_artifacts=[ArtifactRef("diarized_named_corrected", out_path)],
        )
