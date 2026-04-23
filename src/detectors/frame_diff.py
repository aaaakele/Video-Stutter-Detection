from __future__ import annotations

import numpy as np

from src.detectors.base_detector import BaseDetector


class FrameDiffDetector(BaseDetector):
    def __init__(self, threshold: float = 1.2, min_streak: int = 2, **kwargs: object) -> None:
        super().__init__(
            name="frame_diff",
            threshold=threshold,
            min_streak=min_streak,
            **kwargs,
        )

    def score_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
        return float(diff)

    def decide(self, score: float) -> bool:
        return score <= self.threshold
