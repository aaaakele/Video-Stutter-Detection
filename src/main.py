from __future__ import annotations

from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.frame_diff import FrameDiffDetector
from src.detectors.optical_flow import OpticalFlowDetector
from src.detectors.ssim_detector import SSIMDetector
from src.evaluator.metrics import evaluate_events, load_ground_truth
from src.utils.config import RuntimeConfig, load_config
from src.utils.logger import setup_logger
from src.video_reader import VideoReader


DETECTOR_FACTORY = {
    "frame_diff": FrameDiffDetector,
    "ssim": SSIMDetector,
    "optical_flow": OpticalFlowDetector,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_detector_summary(name: str, runtime_sec: float, frame_count: int, event_count: int) -> dict:
    fps = frame_count / runtime_sec if runtime_sec > 0 else 0.0
    complexity_map = {
        "frame_diff": "Low",
        "ssim": "Medium",
        "optical_flow": "High",
    }
    return {
        "method": name,
        "runtime_sec": round(runtime_sec, 4),
        "processing_fps": round(fps, 2),
        "complexity": complexity_map.get(name, "Unknown"),
        "event_count": event_count,
        "time_unit": "second_half_open_interval",
    }


def run(config: RuntimeConfig) -> None:
    logger = setup_logger()
    ensure_dir(config.output_dir)
    logger.info("Starting stutter detection pipeline")
    logger.info("Video path: %s", config.video_path)
    logger.info("Ground truth path: %s", config.gt_path)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Sample every: %s", config.sample_every)

    reader = VideoReader(config.video_path, sample_every=config.sample_every)
    logger.info("Reading video metadata...")
    meta = reader.metadata()
    logger.info(
        "Video metadata loaded: fps=%.3f, frame_count=%s, resolution=%sx%s, duration_ms=%.2f",
        meta.fps,
        meta.frame_count,
        meta.width,
        meta.height,
        meta.duration_ms,
    )

    logger.info("Loading ground truth events...")
    gt_events = load_ground_truth(config.gt_path) if config.gt_path and config.gt_path.exists() else []
    logger.info("Loaded %s ground truth events", len(gt_events))

    all_reports = []
    benchmark_rows = []

    for detector_cfg in config.detectors:
        if not detector_cfg.enabled:
            logger.info("Skipping disabled detector: %s", detector_cfg.name)
            continue
        detector_cls = DETECTOR_FACTORY[detector_cfg.name]
        detector = detector_cls(
            threshold=detector_cfg.threshold,
            min_streak=detector_cfg.min_streak,
            **detector_cfg.extra,
        )
        logger.info(
            "Running detector=%s with threshold=%s, min_streak=%s, extra=%s",
            detector_cfg.name,
            detector_cfg.threshold,
            detector_cfg.min_streak,
            detector_cfg.extra,
        )
        start = time.perf_counter()
        frame_scores, events = detector.detect(reader.iter_frames())
        runtime_sec = time.perf_counter() - start
        logger.info(
            "Detector=%s finished frame scan: frame_scores=%s, events=%s, runtime_sec=%.3f",
            detector.name,
            len(frame_scores),
            len(events),
            runtime_sec,
        )

        payload = {
            "video_meta": asdict(meta),
            "detector": detector.name,
            "time_unit": "second",
            "config": asdict(detector_cfg),
            "frame_scores": [asdict(item) for item in frame_scores],
            "events": [asdict(item) for item in events],
            "summary": build_detector_summary(detector.name, runtime_sec, meta.frame_count, len(events)),
        }

        report_path = config.output_dir / f"{detector.name}_detections.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Detection result written: %s", report_path)

        row = payload["summary"]
        if gt_events:
            logger.info("Evaluating detector=%s against ground truth...", detector.name)
            evaluation = evaluate_events(detector.name, events, gt_events)
            eval_dict = {
                "detector": evaluation.detector,
                "stats": asdict(evaluation.stats),
                "matched_pairs": evaluation.matched_pairs,
            }
            eval_path = config.output_dir / f"{detector.name}_evaluation.json"
            eval_path.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Evaluation result written: %s", eval_path)
            row.update(asdict(evaluation.stats))
            all_reports.append(eval_dict)
        benchmark_rows.append(row)
        logger.info("Detector %s finished with %s events", detector.name, len(events))

    summary = {
        "video_meta": asdict(meta),
        "benchmark": benchmark_rows,
        "evaluations": all_reports,
    }
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Summary written: %s", summary_path)
    logger.info("Stutter detection pipeline completed successfully")


def main() -> None:
    config = load_config()
    run(config)


if __name__ == "__main__":
    main()
