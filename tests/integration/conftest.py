from __future__ import annotations

from pathlib import Path

import pytest
import requests

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.llm import OllamaClient
from mentat_session_logger.models import SessionContext
from mentat_session_logger.prompts import PromptRenderer

OLLAMA_BASE = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    "laptop": {
        "model": "phi3:mini",
        "timeout": 90,
        "description": "CPU-friendly - fits in ~2 GB RAM",
    },
    "rig": {
        "model": "llama3.1:8b",
        "timeout": 180,
        "description": "GPU-accelerated - 8 GB VRAM recommended",
    },
}


# ---------------------------------------------------------------------------
# pytest CLI option
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-profile",
        default="laptop",
        choices=list(PROFILES),
        help="LLM hardware profile: 'laptop' (phi3:mini) or 'rig' (llama3.1:8b).",
    )


# ---------------------------------------------------------------------------
# Shared session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def llm_profile(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--llm-profile"))


@pytest.fixture(scope="session")
def ollama_model(llm_profile: str) -> str:
    return PROFILES[llm_profile]["model"]


@pytest.fixture(scope="session")
def ollama_timeout(llm_profile: str) -> int:
    return int(PROFILES[llm_profile]["timeout"])


@pytest.fixture(scope="session")
def ollama_client(ollama_model: str, ollama_timeout: int) -> OllamaClient:
    """
    Resolve an OllamaClient for the active profile.
    Skips the whole session if Ollama is not running or the model is missing.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE}: {exc}")

    tags = resp.json()
    available: list[str] = [m.get("name", "") for m in tags.get("models", [])]
    if not any(ollama_model in tag for tag in available):
        pytest.skip(
            f"Model '{ollama_model}' not found in Ollama. "
            f"Pull it first:\n\n    ollama pull {ollama_model}\n"
        )

    return OllamaClient(
        endpoint=f"{OLLAMA_BASE}/api/generate",
        model=ollama_model,
        timeout_seconds=ollama_timeout,
    )


@pytest.fixture(scope="session")
def prompt_renderer() -> PromptRenderer:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return PromptRenderer(prompts_dir)


@pytest.fixture()
def session_env(tmp_path: Path):
    """Minimal EnvironmentConfig wired to a temp directory."""
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    return resolver.resolve("local")


@pytest.fixture()
def session_artifacts(session_env, tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(session_env.root, "session_llm_test")
    store.ensure_session_dirs()
    return store


@pytest.fixture()
def session_context(session_env) -> SessionContext:
    return SessionContext(env=session_env, session_id="session_llm_test")
