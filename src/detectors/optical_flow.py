from __future__ import annotations

import cv2
import numpy as np

from src.detectors.base_detector import BaseDetector


class OpticalFlowDetector(BaseDetector):
    def __init__(self, threshold: float = 0.08, min_streak: int = 2, **kwargs: object) -> None:
        super().__init__(
            name="optical_flow",
            threshold=threshold,
            min_streak=min_streak,
            **kwargs,
        )

    def score_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        corners = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=int(self.params.get("max_corners", 200)),
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if corners is None:
            return 0.0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, corners, None)
        if next_points is None or status is None:
            return 0.0

        valid_prev = corners[status.flatten() == 1]
        valid_next = next_points[status.flatten() == 1]
        if len(valid_prev) == 0:
            return 0.0

        displacement = np.linalg.norm(valid_next - valid_prev, axis=1)
        return float(np.mean(displacement))

    def decide(self, score: float) -> bool:
        return score <= self.threshold
