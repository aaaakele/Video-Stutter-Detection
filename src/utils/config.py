from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    name: str
    enabled: bool = True
    threshold: float = 0.0
    min_streak: int = 2
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    video_path: Path
    gt_path: Path | None
    output_dir: Path
    sample_every: int = 1
    detectors: list[DetectorConfig] = field(default_factory=list)


DEFAULT_CONFIG = {
    "video_path": "data/sample.mkv",
    "gt_path": "data/dataset.xlsx",
    "output_dir": "results",
    "sample_every": 1,
    "detectors": [
        {
            "name": "frame_diff",
            "enabled": True,
            "threshold": 0.8,
            "min_streak": 2,
            "extra": {
                "resize_width": 640,
                "min_positive_ratio": 0.6,
                "min_positive_frames": 3,
                "min_event_seconds": 1,
                "max_gap_seconds": 0,
            },
        },
        {
            "name": "ssim",
            "enabled": True,
            "threshold": 0.995,
            "min_streak": 2,
            "extra": {
                "resize_width": 640,
                "min_positive_ratio": 0.55,
                "min_positive_frames": 3,
                "min_event_seconds": 1,
                "max_gap_seconds": 1,
            },
        },
        {
            "name": "optical_flow",
            "enabled": True,
            "threshold": 0.12,
            "min_streak": 2,
            "extra": {
                "resize_width": 640,
                "max_corners": 200,
                "min_positive_ratio": 0.6,
                "min_positive_frames": 3,
                "min_event_seconds": 1,
                "max_gap_seconds": 0,
            },
        },
    ],
}


def load_config(config_path: str | Path | None = None) -> RuntimeConfig:
    raw = DEFAULT_CONFIG
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

    detectors = [DetectorConfig(**item) for item in raw.get("detectors", [])]
    gt_value = raw.get("gt_path")
    return RuntimeConfig(
        video_path=Path(raw["video_path"]),
        gt_path=Path(gt_value) if gt_value else None,
        output_dir=Path(raw["output_dir"]),
        sample_every=int(raw.get("sample_every", 1)),
        detectors=detectors,
    )
