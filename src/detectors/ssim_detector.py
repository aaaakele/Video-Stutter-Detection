from __future__ import annotations

import numpy as np

from src.detectors.base_detector import BaseDetector


class SSIMDetector(BaseDetector):
    def __init__(self, threshold: float = 0.995, min_streak: int = 2, **kwargs: object) -> None:
        super().__init__(
            name="ssim",
            threshold=threshold,
            min_streak=min_streak,
            **kwargs,
        )

    def score_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        prev = prev_frame.astype(np.float64)
        curr = curr_frame.astype(np.float64)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_prev = prev.mean()
        mu_curr = curr.mean()
        sigma_prev = prev.var()
        sigma_curr = curr.var()
        covariance = ((prev - mu_prev) * (curr - mu_curr)).mean()

        numerator = (2 * mu_prev * mu_curr + c1) * (2 * covariance + c2)
        denominator = (mu_prev**2 + mu_curr**2 + c1) * (sigma_prev + sigma_curr + c2)
        if denominator == 0:
            return 1.0
        score = numerator / denominator
        return float(np.clip(score, 0.0, 1.0))

    def decide(self, score: float) -> bool:
        return score >= self.threshold
