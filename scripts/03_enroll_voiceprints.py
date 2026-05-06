from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.voiceprints import SimpleEmbeddingBackend, VoiceprintService

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    args = parser.parse_args()

    env = EnvironmentResolver(Path.cwd()).resolve(args.env)
    service = VoiceprintService(SimpleEmbeddingBackend())
    profiles = service.enroll_environment(env.root)
    logger.info("Enrolled profiles: %s", profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
