from pathlib import Path

from mentat_session_logger.io import write_yaml
from mentat_session_logger.pipeline import load_pipeline_config


def test_pipeline_config_loading(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "configs" / "pipelines"
    pipeline_dir.mkdir(parents=True)
    write_yaml(
        pipeline_dir / "default.yml",
        {
            "pipeline": [
                {"stage": "prepare_audio", "enabled": True},
                {"stage": "transcribe", "enabled": False},
            ]
        },
    )

    cfg = load_pipeline_config(tmp_path, "default")
    assert cfg.name == "default"
    assert len(cfg.stages) == 2
    assert cfg.stages[0].stage == "prepare_audio"
    assert cfg.stages[1].enabled is False
