from __future__ import annotations

from mentat_session_logger.artifacts import ArtifactStore
from mentat_session_logger.io import read_json, write_json
from mentat_session_logger.models import ArtifactRef, SessionContext, StageResult


class DiarizationStage:
    name = "diarize"

    def run(self, context: SessionContext, artifacts: ArtifactStore) -> StageResult:
        whisperx_path = artifacts.raw_file("whisperx_output.json")
        if not whisperx_path.exists():
            raise FileNotFoundError("whisperx_output.json not found")

        payload = read_json(whisperx_path)
        segments = payload.get("segments", [])
        diarized = []
        for segment in segments:
            diarized.append(
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "speaker": str(segment.get("speaker", "SPEAKER_??")),
                    "text": str(segment.get("text", "")).strip(),
                }
            )
        out_path = artifacts.raw_file("diarization.json")
        write_json(out_path, {"session": context.session_id, "segments": diarized})
        return StageResult(
            input_artifacts=[ArtifactRef("whisperx_output", whisperx_path)],
            output_artifacts=[ArtifactRef("diarization", out_path)],
        )
