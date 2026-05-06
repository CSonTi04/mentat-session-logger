from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.campaign_memory import MemoryUpdateProposalStage
from mentat_session_logger.classification import FinalNotebookGenerationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
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
    logger.info("Wrote final outputs in: %s", artifacts.session_root / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
