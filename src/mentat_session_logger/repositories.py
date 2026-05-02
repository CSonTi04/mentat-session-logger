from __future__ import annotations

from pathlib import Path
from typing import Any

from mentat_session_logger.io import read_text, read_yaml, write_text, write_yaml


class SessionRepository:
    def __init__(self, session_root: Path) -> None:
        self.session_root = session_root

    def read_transcript(self, relative_path: str) -> str:
        return read_text(self.session_root / relative_path)

    def write_transcript(self, relative_path: str, content: str) -> None:
        write_text(self.session_root / relative_path, content)


class CampaignMemoryRepository:
    def __init__(self, campaign_context_root: Path) -> None:
        self.root = campaign_context_root

    def read_yaml(self, relative_path: str) -> dict[str, Any]:
        return read_yaml(self.root / relative_path)

    def write_yaml(self, relative_path: str, payload: dict[str, Any]) -> None:
        write_yaml(self.root / relative_path, payload)

    def append_markdown(self, relative_path: str, block: str) -> None:
        path = self.root / relative_path
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        text = previous.rstrip() + "\n\n" + block.strip() + "\n"
        write_text(path, text)
