from pathlib import Path

import pytest

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.campaign_memory import (
    ApprovedMemoryApplyStage,
    MemoryUpdateProposalStage,
)
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.io import write_yaml
from mentat_session_logger.models import SessionContext


def test_memory_proposal_is_generated(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    artifacts = ArtifactStore(env.root, "session_012")
    artifacts.ensure_session_dirs()

    MemoryUpdateProposalStage().run(SessionContext(env=env, session_id="session_012"), artifacts)
    assert artifacts.final_file("memory_update_proposal.yml").exists()


def test_memory_not_applied_without_approval(tmp_path: Path) -> None:
    resolver = EnvironmentResolver(tmp_path)
    resolver.init_env("local")
    env = resolver.resolve("local")
    artifacts = ArtifactStore(env.root, "session_012")
    artifacts.ensure_session_dirs()
    write_yaml(artifacts.final_file("memory_update_proposal.yml"), {"approved": False})

    with pytest.raises(PermissionError):
        ApprovedMemoryApplyStage().run(SessionContext(env=env, session_id="session_012"), artifacts)
