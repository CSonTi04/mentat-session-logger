from __future__ import annotations

import argparse
from pathlib import Path

from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.voiceprints import SimpleEmbeddingBackend, VoiceprintService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    args = parser.parse_args()

    env = EnvironmentResolver(Path.cwd()).resolve(args.env)
    service = VoiceprintService(SimpleEmbeddingBackend())
    profiles = service.enroll_environment(env.root)
    print(f"Enrolled profiles: {profiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
