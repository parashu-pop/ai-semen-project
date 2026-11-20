from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "dg_exp1"
    / "weights"
    / "last.pt"
)
TRACKER_CONFIG = Path(__file__).resolve().parent / "bytetrack_sperm.yaml"


def _conditions(seed: int) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    random = np.random.default_rng(seed)

    def add_noise(frame: np.ndarray) -> np.ndarray:
        noise = random.normal(0.0, 12.0, frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return [
        ("original", lambda frame: frame),
        (
            "low_brightness",
            lambda frame: cv2.convertScaleAbs(frame, alpha=1.0, beta=-40),
        ),
        (
            "high_brightness",
            lambda frame: cv2.convertScaleAbs(frame, alpha=1.0, beta=40),
        ),
        (
            "gaussian_blur",
            lambda frame: cv2.GaussianBlur(frame, (5, 5), 0),
        ),
        ("gaussian_noise", add_noise),
        (
            "low_contrast",
            lambda frame: cv2.convertScaleAbs(frame, alpha=0.7, beta=0),
        ),
        (
            "high_contrast",
            lambda frame: cv2.convertScaleAbs(frame, alpha=1.3, beta=0),
        ),
    ]


def run_robustness_test(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    max_frames: int = 300,
    seed: int = 42,
) -> list[dict[str, int | float | str]]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    rows: list[dict[str, int | float | str]] = []
    for condition_name, transform in _conditions(seed):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        model = YOLO(str(model_path))
        frame_count = 0
        detection_count = 0
        confidence_sum = 0.0
        track_frames: dict[int, set[int]] = defaultdict(set)
        while frame_count < max_frames:
            success, frame = capture.read()
            if not success:
                break

            transformed = transform(frame)
            results = model.track(
                transformed,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                conf=0.25,
                iou=0.5,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                detection_count += len(boxes)
                confidence_sum += float(boxes.conf.sum().item())
                if boxes.id is not None:
                    for track_id in boxes.id.int().cpu().tolist():
                        track_frames[int(track_id)].add(frame_count)
            frame_count += 1

        capture.release()
        track_lengths = [len(frames) for frames in track_frames.values()]
        rows.append(
            {
                "condition": condition_name,
                "frames": frame_count,
                "detections": detection_count,
                "detections_per_frame": (
                    detection_count / frame_count if frame_count else 0.0
                ),
                "mean_confidence": (
                    confidence_sum / detection_count if detection_count else 0.0
                ),
                "unique_track_ids": len(track_frames),
                "mean_track_observations": (
                    sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
                ),
                "metric_note": (
                    "Detection/track-count proxies only; no ground-truth IDs, "
                    "so ID-switch accuracy is not measured."
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLO tracking proxies under video brightness/noise variations."
    )
    parser.add_argument("--video", type=Path, default=PROJECT_DIR / "temp_video.mp4")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "sperm_robustness_report.csv",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_robustness_test(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
