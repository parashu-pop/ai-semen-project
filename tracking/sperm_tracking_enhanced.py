from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .motility_analysis import (
    Detection,
    classify_motility,
    compute_track_metrics,
    remove_duplicate_detections,
)


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
MICROMETRES_PER_PIXEL: float | None = None
TRACKER_CONFIG = Path(__file__).resolve().parent / "bytetrack_sperm.yaml"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLASS_COLORS = {
    "Progressive": (0, 200, 0),
    "Non-Progressive": (0, 165, 255),
    "Immotile": (0, 0, 255),
    "Insufficient data": (160, 160, 160),
}


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    color = cv2.cvtColor(
        np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def _live_tracking_preview(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    preview = frame.copy()
    for detection in detections:
        track_id = detection["track_id"]
        color = _track_color(track_id)
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            preview,
            f"ID:{track_id}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    empty_fieldnames: list[str],
) -> None:
    fieldnames = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else empty_fieldnames
    )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _annotate_video(
    video_path: Path,
    output_path: Path,
    frame_detections: dict[int, list[Detection]],
    classifications: dict[int, str],
    fps: float,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create annotated video: {output_path}")

    histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame_index in range(frame_count):
        success, frame = capture.read()
        if not success:
            break

        for detection in frame_detections.get(frame_index, []):
            track_id = detection["track_id"]
            x = int(round(detection["x"]))
            y = int(round(detection["y"]))
            box = detection["box"]
            category = classifications[track_id]
            category_color = CLASS_COLORS[category]
            trail_color = _track_color(track_id)
            history = histories[track_id]
            history.append((x, y))

            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), category_color, 2)
            cv2.putText(
                frame,
                f"ID:{track_id} {category}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                category_color,
                1,
                cv2.LINE_AA,
            )
            for previous, current in zip(history[-20:-1], history[-19:]):
                cv2.line(frame, previous, current, trail_color, 2)
            if len(history) == 1:
                cv2.circle(frame, history[0], 5, trail_color, -1)
            else:
                cv2.arrowedLine(
                    frame,
                    history[-2],
                    history[-1],
                    trail_color,
                    2,
                    tipLength=0.25,
                )
                cv2.circle(frame, history[0], 4, (255, 255, 255), -1)
                cv2.circle(frame, history[-1], 4, trail_color, -1)

        cv2.rectangle(frame, (8, 8), (width - 8, 34), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "Trajectory analysis | pixel-based units",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    capture.release()
    writer.release()


def _write_heatmap(
    video_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    success, first_frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Unable to read first frame from {video_path}")

    smoothed = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8)
    scaled = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(first_frame, 0.55, colored, 0.45, 0)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Unable to save heatmap: {output_path}")


def _write_preview_gif(
    video_path: Path,
    output_path: Path,
    frame_count: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open annotated video: {video_path}")

    sample_count = min(20, frame_count)
    sample_indices = np.linspace(
        0, max(0, frame_count - 1), num=sample_count, dtype=int
    )
    frames = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb)
        preview.thumbnail((480, 360))
        frames.append(preview.quantize(colors=128))
    capture.release()

    if not frames:
        raise RuntimeError(f"Unable to decode preview frames from {video_path}")
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
        optimize=True,
    )


def run_analysis(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    progressive_speed_px_s: float = 20.0,
    motile_speed_px_s: float = 6.7,
    max_frames: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
    micrometres_per_pixel = MICROMETRES_PER_PIXEL
    if micrometres_per_pixel is not None and micrometres_per_pixel <= 0:
        raise ValueError("Internal microscope calibration must be greater than zero")
    if progressive_speed_px_s < motile_speed_px_s:
        raise ValueError(
            "Progressive speed threshold must be at least the motile threshold"
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    metadata_frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_frame_count = (
        int(metadata_frame_count_value)
        if np.isfinite(metadata_frame_count_value) and metadata_frame_count_value > 0
        else 0
    )
    fps = video_fps
    if (
        width <= 0
        or height <= 0
        or not np.isfinite(fps)
        or fps <= 0
    ):
        capture.release()
        raise ValueError(
            "Video metadata must include valid width, height, and FPS."
        )

    model = YOLO(str(model_path))
    frame_detections: dict[int, list[Detection]] = {}
    track_points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    heatmap = np.zeros((height, width), dtype=np.float32)
    frame_index = 0
    total_detections = 0

    while True:
        success, frame = capture.read()
        if not success or (max_frames is not None and frame_index >= max_frames):
            break

        results = model.track(
            frame,
            persist=True,
            tracker=str(TRACKER_CONFIG),
            conf=0.25,
            iou=0.5,
            verbose=False,
        )
        boxes = results[0].boxes
        detections: list[Detection] = []
        if boxes is not None:
            total_detections += len(boxes)
        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().numpy()
            centers = boxes.xywh.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for box, center, track_id, confidence in zip(
                coordinates, centers, track_ids, confidences
            ):
                detections.append(
                    Detection(
                        frame=frame_index,
                        time_s=frame_index / fps,
                        track_id=int(track_id),
                        x=float(center[0]),
                        y=float(center[1]),
                        confidence=float(confidence),
                        box=tuple(float(value) for value in box),
                    )
                )

        frame_detections[frame_index] = remove_duplicate_detections(detections)
        for detection in frame_detections[frame_index]:
            track_id = detection["track_id"]
            x = detection["x"]
            y = detection["y"]
            track_points[track_id].append((frame_index, x, y))
            heat_x = int(np.clip(round(x), 0, width - 1))
            heat_y = int(np.clip(round(y), 0, height - 1))
            heatmap[heat_y, heat_x] += 1.0
        frame_index += 1
        emit_update = (
            frame_index % max(1, int(fps)) == 0
            or (
                metadata_frame_count > 0
                and frame_index >= metadata_frame_count
            )
            or (max_frames is not None and frame_index >= max_frames)
        )
        if progress_callback is not None and emit_update:
            progress_callback(frame_index, metadata_frame_count)
        if preview_callback is not None and emit_update:
            preview_callback(
                _live_tracking_preview(
                    frame,
                    frame_detections[frame_index - 1],
                )
            )

    capture.release()
    if frame_index == 0:
        raise RuntimeError("No video frames were processed")

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    classifications: dict[int, str] = {}
    unit_label = "pixel-based"
    if micrometres_per_pixel is not None:
        unit_label = f"calibrated at {micrometres_per_pixel:g} µm/pixel"

    for track_id, points in sorted(track_points.items()):
        metrics = compute_track_metrics(points, fps, micrometres_per_pixel)
        classification = classify_motility(
            metrics,
            progressive_speed_px_s=progressive_speed_px_s,
            motile_speed_px_s=motile_speed_px_s,
        )
        classifications[track_id] = classification
        metric_rows.append(
            {
                "track_id": track_id,
                "classification": classification,
                "measurement_units": unit_label,
                "fps": fps,
                "progressive_threshold_px_s": progressive_speed_px_s,
                "motile_threshold_px_s": motile_speed_px_s,
                **metrics,
            }
        )

    track_rows = [
        {
            "frame": detection["frame"],
            "time_s": round(detection["time_s"], 6),
            "track_id": detection["track_id"],
            "x_px": round(detection["x"], 3),
            "y_px": round(detection["y"], 3),
            "confidence": round(detection["confidence"], 6),
            "x1_px": round(detection["box"][0], 3),
            "y1_px": round(detection["box"][1], 3),
            "x2_px": round(detection["box"][2], 3),
            "y2_px": round(detection["box"][3], 3),
        }
        for frame_rows in frame_detections.values()
        for detection in frame_rows
    ]

    tracks_csv = output_dir / "sperm_tracks_enhanced.csv"
    metrics_csv = output_dir / "sperm_track_metrics_enhanced.csv"
    video_output = output_dir / "sperm_tracking_enhanced.mp4"
    preview_output = output_dir / "sperm_tracking_preview.gif"
    heatmap_output = output_dir / "sperm_movement_heatmap.png"
    summary_json = output_dir / "sperm_motility_summary_enhanced.json"

    _write_csv(
        tracks_csv,
        track_rows,
        [
            "frame",
            "time_s",
            "track_id",
            "x_px",
            "y_px",
            "confidence",
            "x1_px",
            "y1_px",
            "x2_px",
            "y2_px",
        ],
    )
    _write_csv(
        metrics_csv,
        [
            row
            for row in metric_rows
            if row["classification"] != "Insufficient data"
        ],
        [
            "track_id",
            "classification",
            "measurement_units",
            "fps",
            "progressive_threshold_px_s",
            "motile_threshold_px_s",
            "observed_points",
            "duration_s",
            "VCL_px_s",
            "VSL_px_s",
            "VAP_px_s",
            "LIN",
            "STR",
            "ALH_px",
            "BCF_hz_estimate",
        ]
        + (
            ["VCL_um_s", "VSL_um_s", "VAP_um_s", "ALH_um"]
            if micrometres_per_pixel is not None
            else []
        ),
    )
    _annotate_video(
        video_path,
        video_output,
        frame_detections,
        classifications,
        fps,
        width,
        height,
        frame_index,
    )
    _write_preview_gif(video_output, preview_output, frame_index)
    _write_heatmap(video_path, heatmap, heatmap_output)

    classification_counts = {
        category: sum(
            classification == category
            for classification in classifications.values()
        )
        for category in ("Progressive", "Non-Progressive", "Immotile")
    }
    classified_track_count = sum(
        classification_counts[category]
        for category in ("Progressive", "Non-Progressive", "Immotile")
    )
    motility_percentages = {
        category: (
            100.0 * classification_counts[category] / classified_track_count
            if classified_track_count
            else None
        )
        for category in ("Progressive", "Non-Progressive", "Immotile")
    }
    motility_percentages["Total motile"] = (
        100.0
        * (
            classification_counts["Progressive"]
            + classification_counts["Non-Progressive"]
        )
        / classified_track_count
        if classified_track_count
        else None
    )

    track_consistency = []
    for points in track_points.values():
        observed_frames = {point[0] for point in points}
        span = max(observed_frames) - min(observed_frames) + 1
        if span:
            track_consistency.append(len(observed_frames) / span)

    summary_metric_keys = (
        "VCL_px_s",
        "VSL_px_s",
        "VAP_px_s",
        "LIN",
        "STR",
        "ALH_px",
        "BCF_hz_estimate",
    )
    average_metrics = {}
    for key in summary_metric_keys:
        values = [
            float(row[key])
            for row in metric_rows
            if np.isfinite(float(row[key]))
        ]
        average_metrics[key] = float(np.mean(values)) if values else None

    summary: dict[str, Any] = {
        "video": str(video_path),
        "model": str(model_path),
        "fps": fps,
        "width": width,
        "height": height,
        "metadata_frame_count": metadata_frame_count,
        "measurement_units": unit_label,
        "frames_processed": frame_index,
        "detections": total_detections,
        "tracked_detection_observations": len(track_rows),
        "unique_tracks": len(track_points),
        "tracking_consistency_pct": (
            100.0 * float(np.mean(track_consistency))
            if track_consistency
            else None
        ),
        "classification_counts": classification_counts,
        "classified_track_count": classified_track_count,
        "motility_percentages": motility_percentages,
        "average_metrics": average_metrics,
        "outputs": {
            "tracks_csv": str(tracks_csv),
            "metrics_csv": str(metrics_csv),
            "video": str(video_output),
            "preview": str(preview_output),
            "heatmap": str(heatmap_output),
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated sperm tracking and motility analysis."
    )
    parser.add_argument("--video", type=Path, default=PROJECT_DIR / "temp_video.mp4")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_analysis(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
