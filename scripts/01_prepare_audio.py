from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.audio import AudioCommandRunner
from mentat_session_logger.environments import EnvironmentResolver

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
    parser.add_argument("--input", required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    workspace = Path.cwd()
    env = EnvironmentResolver(workspace).resolve(args.env)
    artifacts = ArtifactStore(env.root, args.session)
    artifacts.ensure_session_dirs()

    runner = AudioCommandRunner()
    runner.validate()

    input_audio = Path(args.input)
    prepared = artifacts.audio_file(f"{args.session}_16k.wav")
    runner.to_mono_16k(input_audio, prepared)
    logger.info("Prepared audio: %s", prepared)

    if args.normalize:
        normalized = artifacts.audio_file(f"{args.session}_16k_norm.wav")
        runner.normalize_loudness(prepared, normalized)
        logger.info("Normalized audio: %s", normalized)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
