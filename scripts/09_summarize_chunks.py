from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.classification import ChunkSummarizationStage
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.llm import OllamaClient
from mentat_session_logger.models import SessionContext
from mentat_session_logger.prompts import PromptRenderer

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

    workspace = Path.cwd()
    env = EnvironmentResolver(workspace).resolve(args.env)
    context = SessionContext(env=env, session_id=args.session)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    llm = OllamaClient(endpoint=env.llm_endpoint, model=env.llm_model)
    prompts = PromptRenderer(workspace / "prompts")
    ChunkSummarizationStage(llm, prompts).run(context, artifacts)
    logger.info("Wrote chunk summaries in: %s", artifacts.session_root / "chunk_summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
