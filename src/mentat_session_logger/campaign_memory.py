from __future__ import annotations

from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_text, read_yaml, write_text, write_yaml
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult


class MemoryUpdateProposalStage:
    name = "propose_memory_update"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        canon_delta = artifacts.final_file("canon_delta.md")
        session_notebook = artifacts.final_file("session_notebook.md")
        source_summary = read_text(session_notebook) if session_notebook.exists() else ""
        delta = read_text(canon_delta) if canon_delta.exists() else ""

        payload = {
            "session": context.session_id,
            "approved": False,
            "new_canon_candidates": [
                {
                    "fact": "Review generated session notebook for canon candidates.",
                    "confidence": "medium",
                    "source_time": "00:00:00-00:00:00",
                    "suggested_destination": "canon_uncertain.md",
                }
            ],
            "possible_contradictions": [],
            "notes": {
                "notebook_excerpt": source_summary[:500],
                "canon_delta_excerpt": delta[:500],
            },
        }
        out_path = artifacts.final_file("memory_update_proposal.yml")
        write_yaml(out_path, payload)
        return StageResult(
            input_artifacts=[ArtifactRef("session_notebook", session_notebook)],
            output_artifacts=[ArtifactRef("memory_update_proposal", out_path)],
        )


class ApprovedMemoryApplyStage:
    name = "apply_approved_memory_update"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        proposal = artifacts.final_file("memory_update_proposal.yml")
        if not proposal.exists():
            raise FileNotFoundError("memory_update_proposal.yml not found")

        payload = read_yaml(proposal)
        if payload.get("approved") is not True:
            raise PermissionError(
                "Proposal is not approved. Set approved: true after human review."
            )

        campaign_root = context.env.campaign_context_dir
        approved_path = campaign_root / "canon_log" / "canon_approved.md"
        uncertain_path = campaign_root / "canon_log" / "canon_uncertain.md"

        facts = payload.get("new_canon_candidates", [])
        approved_lines: list[str] = []
        uncertain_lines: list[str] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact", "")).strip()
            destination = str(item.get("suggested_destination", "canon_uncertain.md"))
            if not fact:
                continue
            line = f"- {fact}"
            if destination == "canon_approved.md":
                approved_lines.append(line)
            else:
                uncertain_lines.append(line)

        if approved_lines:
            _append_markdown(approved_path, "\n".join(approved_lines))
        if uncertain_lines:
            _append_markdown(uncertain_path, "\n".join(uncertain_lines))

        timeline = campaign_root / "timeline.md"
        _append_markdown(timeline, f"- {context.session_id}: memory update applied")

        session_summary_path = (
            campaign_root / "previous_session_summaries" / f"{context.session_id}.md"
        )
        notebook = artifacts.final_file("session_notebook.md")
        summary_text = read_text(notebook) if notebook.exists() else f"# {context.session_id}\n"
        write_text(session_summary_path, summary_text)

        return StageResult(
            input_artifacts=[ArtifactRef("approved_proposal", proposal)],
            output_artifacts=[
                ArtifactRef("canon_approved", approved_path),
                ArtifactRef("canon_uncertain", uncertain_path),
                ArtifactRef("timeline", timeline),
                ArtifactRef("session_summary", session_summary_path),
            ],
        )


def _append_markdown(path: Path, text: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    write_text(path, current.rstrip() + "\n" + text.strip() + "\n")
