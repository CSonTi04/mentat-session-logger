from pathlib import Path

import pytest

from mentat_session_logger.environments import EnvironmentResolver


def test_init_env_does_not_overwrite_without_force(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    first = resolver.init_env("local")
    assert first.exists()

    with pytest.raises(FileExistsError):
        resolver.init_env("local", force=False)


def test_init_env_with_force_succeeds(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    second = resolver.init_env("local", force=True)
    assert second.exists()


def test_resolve_env_reads_defaults(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    assert env.name == "local"
    assert env.asr_language == "hu"
    assert env.min_speakers == 4
