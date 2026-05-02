from __future__ import annotations

import argparse
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.campaign_memory import MemoryUpdateProposalStage
from mentat_session_logger.classification import FinalNotebookGenerationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    env = EnvironmentResolver(Path.cwd()).resolve(args.env)
    context = SessionContext(env=env, session_id=args.session)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    FinalNotebookGenerationStage().run(context, artifacts)
    MemoryUpdateProposalStage().run(context, artifacts)
    print(f"Wrote final outputs in: {artifacts.session_root / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
