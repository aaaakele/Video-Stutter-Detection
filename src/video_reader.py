from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: float


class VideoReader:
    def __init__(self, video_path: str | Path, sample_every: int = 1) -> None:
        self.video_path = str(video_path)
        self.sample_every = max(1, sample_every)

    def metadata(self) -> VideoMeta:
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()

        duration_ms = (frame_count / fps * 1000.0) if fps else 0.0
        return VideoMeta(
            path=self.video_path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_ms=duration_ms,
        )

    def iter_frames(self) -> Iterator[FrameRecord]:
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % self.sample_every == 0:
                    timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    yield FrameRecord(
                        frame_index=frame_index,
                        timestamp_ms=float(timestamp_ms),
                        image=frame,
                    )
                frame_index += 1
        finally:
            capture.release()
