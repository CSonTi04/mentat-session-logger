from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mentat_session_logger.io import ensure_dir, read_yaml, write_yaml
from mentat_session_logger.models import EnvironmentConfig


@dataclass(frozen=True)
class EnvironmentResolver:
    workspace_root: Path

    def resolve(self, env_name: str) -> EnvironmentConfig:
        env_root = self.workspace_root / "envs" / env_name
        config_path = env_root / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Environment config not found: {config_path}")

        cfg = read_yaml(config_path)
        paths = cfg.get("paths", {})
        defaults = cfg.get("defaults", {})
        llm = cfg.get("llm", {})

        campaign_context_dir = env_root / str(paths.get("campaign_context", "campaign_context"))
        voiceprints_dir = env_root / str(paths.get("voiceprints", "voiceprints"))
        sessions_dir = env_root / str(paths.get("sessions", "sessions"))
        profiles_dir = env_root / str(paths.get("profiles", "profiles"))

        return EnvironmentConfig(
            name=str(cfg.get("name", env_name)),
            root=env_root,
            campaign_context_dir=campaign_context_dir,
            voiceprints_dir=voiceprints_dir,
            sessions_dir=sessions_dir,
            profiles_dir=profiles_dir,
            default_pipeline=str(defaults.get("pipeline", "default")),
            output_language=str(defaults.get("output_language", "hu")),
            asr_language=str(defaults.get("asr_language", "hu")),
            min_speakers=int(defaults.get("min_speakers", 4)),
            max_speakers=int(defaults.get("max_speakers", 8)),
            llm_provider=str(llm.get("provider", "ollama")),
            llm_endpoint=str(llm.get("endpoint", "http://localhost:11434/api/generate")),
            llm_model=str(llm.get("model", "llama3.1:8b")),
        )

    def init_env(self, env_name: str, force: bool = False) -> Path:
        env_root = self.workspace_root / "envs" / env_name
        if env_root.exists() and not force:
            raise FileExistsError(f"Environment already exists: {env_root}")

        ensure_dir(env_root)
        for rel in ["campaign_context", "voiceprints", "sessions", "profiles"]:
            ensure_dir(env_root / rel)

        config_path = env_root / "config.yml"
        if force or not config_path.exists():
            write_yaml(
                config_path,
                {
                    "name": env_name,
                    "paths": {
                        "campaign_context": "campaign_context",
                        "voiceprints": "voiceprints",
                        "sessions": "sessions",
                        "profiles": "profiles",
                    },
                    "defaults": {
                        "pipeline": "default",
                        "output_language": "hu",
                        "asr_language": "hu",
                        "min_speakers": 4,
                        "max_speakers": 8,
                    },
                    "llm": {
                        "provider": "ollama",
                        "endpoint": "http://localhost:11434/api/generate",
                        "model": "llama3.1:8b",
                    },
                },
            )

        return env_root
