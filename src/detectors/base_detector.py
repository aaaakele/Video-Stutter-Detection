from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Iterable

import cv2
import numpy as np

from src.video_reader import FrameRecord


@dataclass
class FrameScore:
    frame_index: int
    timestamp_ms: float
    timestamp_sec: int
    score: float
    is_stutter: bool


@dataclass
class StutterEvent:
    start_frame: int
    end_frame: int
    start_sec: int
    end_sec: int
    duration_sec: int
    severity: float
    evidence: dict[str, Any]


class BaseDetector(ABC):
    def __init__(self, name: str, threshold: float, min_streak: int = 2, **kwargs: Any) -> None:
        self.name = name
        self.threshold = threshold
        self.min_streak = min_streak
        self.params = kwargs
        self.logger = logging.getLogger(name)

    @abstractmethod
    def score_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def decide(self, score: float) -> bool:
        raise NotImplementedError

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        resize_width = self.params.get("resize_width")
        if resize_width and frame.shape[1] > resize_width:
            ratio = resize_width / frame.shape[1]
            new_size = (resize_width, int(frame.shape[0] * ratio))
            frame = cv2.resize(frame, new_size)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect(self, frames: Iterable[FrameRecord]) -> tuple[list[FrameScore], list[StutterEvent]]:
        iterator = iter(frames)
        first = next(iterator, None)
        if first is None:
            return [], []

        prev_processed = self.preprocess(first.image)
        frame_scores: list[FrameScore] = []

        for pair_index, current in enumerate(iterator, start=1):
            curr_processed = self.preprocess(current.image)
            score = self.score_pair(prev_processed, curr_processed)
            is_stutter = self.decide(score)
            frame_scores.append(
                FrameScore(
                    frame_index=current.frame_index,
                    timestamp_ms=current.timestamp_ms,
                    timestamp_sec=int(current.timestamp_ms // 1000),
                    score=float(score),
                    is_stutter=is_stutter,
                )
            )
            prev_record = current
            prev_processed = curr_processed

            if pair_index % 300 == 0:
                self.logger.info(
                    "Processed %s frame pairs for detector=%s, latest_frame=%s, latest_score=%.6f",
                    pair_index,
                    self.name,
                    current.frame_index,
                    float(score),
                )

        events = self.merge_scores_to_events(frame_scores)
        return frame_scores, events

    def merge_scores_to_events(self, scores: list[FrameScore]) -> list[StutterEvent]:
        second_buckets: dict[int, list[FrameScore]] = {}
        for item in scores:
            second_buckets.setdefault(item.timestamp_sec, []).append(item)

        second_flags: list[tuple[int, bool, float, int, int]] = []
        min_positive_ratio = float(self.params.get("min_positive_ratio", 0.5))
        min_positive_frames = int(self.params.get("min_positive_frames", 2))

        for second in sorted(second_buckets):
            bucket = second_buckets[second]
            positive_count = sum(1 for item in bucket if item.is_stutter)
            mean_score = float(sum(item.score for item in bucket) / len(bucket))
            positive_ratio = positive_count / len(bucket)
            is_stutter_second = positive_count >= min_positive_frames and positive_ratio >= min_positive_ratio
            second_flags.append((second, is_stutter_second, mean_score, positive_count, len(bucket)))

        events: list[StutterEvent] = []
        streak: list[tuple[int, bool, float, int, int]] = []
        min_event_seconds = max(1, int(self.params.get("min_event_seconds", 1)))
        max_gap_seconds = max(0, int(self.params.get("max_gap_seconds", 0)))

        def flush() -> None:
            nonlocal streak
            if len(streak) >= min_event_seconds:
                start_sec = streak[0][0]
                end_sec = streak[-1][0]
                severity = float(sum(item[2] for item in streak) / len(streak))
                positive_frames = sum(item[3] for item in streak)
                total_frames = sum(item[4] for item in streak)
                events.append(
                    StutterEvent(
                        start_frame=0,
                        end_frame=0,
                        start_sec=start_sec,
                        end_sec=end_sec + 1,
                        duration_sec=(end_sec + 1) - start_sec,
                        severity=severity,
                        evidence={
                            "seconds": len(streak),
                            "interval_type": "half_open",
                            "threshold": self.threshold,
                            "positive_frames": positive_frames,
                            "total_frames": total_frames,
                            "min_positive_ratio": min_positive_ratio,
                            "min_positive_frames": min_positive_frames,
                        },
                    )
                )
            streak = []

        previous_second: int | None = None
        for item in second_flags:
            second, is_stutter_second, _, _, _ = item
            if is_stutter_second:
                if streak and previous_second is not None and second - previous_second > max_gap_seconds + 1:
                    flush()
                streak.append(item)
                previous_second = second
            else:
                flush()
                previous_second = None
        flush()
        return events
