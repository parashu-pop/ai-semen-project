from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict


Point = tuple[int, float, float]


class Detection(TypedDict):
    frame: int
    time_s: float
    track_id: int
    x: float
    y: float
    confidence: float
    box: tuple[float, float, float, float]


def intersection_over_union(
    first: Sequence[float], second: Sequence[float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def remove_duplicate_detections(
    detections: list[Detection], iou_threshold: float = 0.85
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(
        detections, key=lambda item: float(item["confidence"]), reverse=True
    ):
        box = detection["box"]
        if any(
            intersection_over_union(box, previous["box"]) >= iou_threshold
            for previous in kept
        ):
            continue
        kept.append(detection)
    return kept


def _moving_average(
    points: list[tuple[float, float]], radius: int = 2
) -> list[tuple[float, float]]:
    smoothed = []
    for index in range(len(points)):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        window = points[start:end]
        smoothed.append(
            (
                sum(point[0] for point in window) / len(window),
                sum(point[1] for point in window) / len(window),
            )
        )
    return smoothed


def compute_track_metrics(
    points: Sequence[Point],
    fps: float,
    micrometres_per_pixel: float | None = None,
) -> dict[str, float | int]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if micrometres_per_pixel is not None and micrometres_per_pixel <= 0:
        raise ValueError("micrometres_per_pixel must be greater than zero")

    by_frame: dict[int, tuple[float, float]] = {}
    for frame, x, y in points:
        by_frame[int(frame)] = (float(x), float(y))
    ordered = sorted(by_frame.items())
    observed_points = len(ordered)
    if observed_points < 2:
        unavailable = float("nan")
        return {
            "observed_points": observed_points,
            "duration_s": 0.0,
            "VCL_px_s": unavailable,
            "VSL_px_s": unavailable,
            "VAP_px_s": unavailable,
            "LIN": unavailable,
            "STR": unavailable,
            "ALH_px": unavailable,
            "BCF_hz_estimate": unavailable,
        }

    frames = [item[0] for item in ordered]
    raw_points = [item[1] for item in ordered]
    duration_s = (frames[-1] - frames[0]) / fps
    if duration_s <= 0:
        unavailable = float("nan")
        return {
            "observed_points": observed_points,
            "duration_s": 0.0,
            "VCL_px_s": unavailable,
            "VSL_px_s": unavailable,
            "VAP_px_s": unavailable,
            "LIN": unavailable,
            "STR": unavailable,
            "ALH_px": unavailable,
            "BCF_hz_estimate": unavailable,
        }

    path_length = sum(
        math.dist(previous, current)
        for previous, current in zip(raw_points, raw_points[1:])
    )
    straight_displacement = math.dist(raw_points[0], raw_points[-1])
    smoothed = _moving_average(raw_points)
    average_path = [raw_points[0], *smoothed[1:-1], raw_points[-1]]
    average_path_length = sum(
        math.dist(previous, current)
        for previous, current in zip(average_path, average_path[1:])
    )

    vcl = path_length / duration_s
    vsl = straight_displacement / duration_s
    vap = average_path_length / duration_s
    linearity = vsl / vcl if vcl else float("nan")
    straightness = vsl / vap if vap else float("nan")

    lateral_deviations = []
    for index in range(1, len(raw_points) - 1):
        tangent_x = smoothed[index + 1][0] - smoothed[index - 1][0]
        tangent_y = smoothed[index + 1][1] - smoothed[index - 1][1]
        tangent_length = math.hypot(tangent_x, tangent_y)
        residual_x = raw_points[index][0] - smoothed[index][0]
        residual_y = raw_points[index][1] - smoothed[index][1]
        if tangent_length:
            lateral = (
                tangent_x * residual_y - tangent_y * residual_x
            ) / tangent_length
        else:
            lateral = math.hypot(residual_x, residual_y)
        lateral_deviations.append(lateral)

    alh = (
        2.0
        * sum(abs(value) for value in lateral_deviations)
        / len(lateral_deviations)
        if lateral_deviations
        else float("nan")
    )

    nonzero_signs = [1 if value > 0 else -1 for value in lateral_deviations if value]
    if len(nonzero_signs) >= 3:
        crossings = sum(
            previous != current
            for previous, current in zip(nonzero_signs, nonzero_signs[1:])
        )
        bcf = crossings / (2.0 * duration_s) if crossings >= 4 else float("nan")
    else:
        bcf = float("nan")

    metrics: dict[str, float | int] = {
        "observed_points": observed_points,
        "duration_s": duration_s,
        "VCL_px_s": vcl,
        "VSL_px_s": vsl,
        "VAP_px_s": vap,
        "LIN": linearity,
        "STR": straightness,
        "ALH_px": alh,
        "BCF_hz_estimate": bcf,
    }
    if micrometres_per_pixel is not None:
        metrics.update(
            {
                "VCL_um_s": vcl * micrometres_per_pixel,
                "VSL_um_s": vsl * micrometres_per_pixel,
                "VAP_um_s": vap * micrometres_per_pixel,
                "ALH_um": alh * micrometres_per_pixel,
            }
        )
    return metrics


def classify_motility(
    metrics: dict[str, float | int],
    progressive_speed_px_s: float = 20.0,
    motile_speed_px_s: float = 6.7,
    minimum_points: int = 5,
) -> str:
    if metrics["observed_points"] < minimum_points:
        return "Insufficient data"

    vcl = float(metrics["VCL_px_s"])
    vsl = float(metrics["VSL_px_s"])
    straightness = float(metrics["STR"])
    if vcl < motile_speed_px_s:
        return "Immotile"
    if vsl >= progressive_speed_px_s and straightness >= 0.5:
        return "Progressive"
    return "Non-Progressive"
