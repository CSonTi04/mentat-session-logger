from __future__ import annotations

import argparse
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.environments import EnvironmentResolver
from mentat_session_logger.models import SessionContext
from mentat_session_logger.voiceprints import SpeakerMatchingStage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    env = EnvironmentResolver(Path.cwd()).resolve(args.env)
    context = SessionContext(env=env, session_id=args.session)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()
    SpeakerMatchingStage().run(context, artifacts)
    print(f"Wrote: {artifacts.map_file('speaker_match_suggestions.yml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
