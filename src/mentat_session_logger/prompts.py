from __future__ import annotations

from pathlib import Path

from mentat_session_logger.io import read_text


class PromptRenderer:
    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir

    def render(self, prompt_name: str, context: dict[str, str]) -> str:
        template = read_text(self.prompts_dir / prompt_name)
        rendered = template
        for key, value in context.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered
