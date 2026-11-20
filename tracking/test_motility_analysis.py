import unittest
import math

from tracking.motility_analysis import (
    classify_motility,
    compute_track_metrics,
    remove_duplicate_detections,
)


class MotilityAnalysisTests(unittest.TestCase):
    def test_straight_track_metrics_and_optional_calibration(self):
        points = [(frame, frame * 2.0, 0.0) for frame in range(5)]

        metrics = compute_track_metrics(
            points,
            fps=10.0,
            micrometres_per_pixel=0.5,
        )

        self.assertAlmostEqual(metrics["VCL_px_s"], 20.0)
        self.assertAlmostEqual(metrics["VSL_px_s"], 20.0)
        self.assertAlmostEqual(metrics["LIN"], 1.0)
        self.assertAlmostEqual(metrics["STR"], 1.0)
        self.assertAlmostEqual(metrics["VSL_um_s"], 10.0)

    def test_elapsed_time_uses_frame_gaps(self):
        metrics = compute_track_metrics(
            [(0, 0.0, 0.0), (2, 4.0, 0.0)],
            fps=2.0,
        )

        self.assertAlmostEqual(metrics["duration_s"], 1.0)
        self.assertAlmostEqual(metrics["VCL_px_s"], 4.0)

    def test_classification_requires_enough_observations(self):
        metrics = compute_track_metrics(
            [(0, 0.0, 0.0), (1, 5.0, 0.0)],
            fps=10.0,
        )

        self.assertEqual(classify_motility(metrics), "Insufficient data")

    def test_unmeasurable_features_are_not_reported_as_zero(self):
        one_point = compute_track_metrics([(0, 0.0, 0.0)], fps=10.0)
        two_points = compute_track_metrics(
            [(0, 0.0, 0.0), (1, 2.0, 0.0)],
            fps=10.0,
        )

        self.assertTrue(math.isnan(one_point["VCL_px_s"]))
        self.assertTrue(math.isnan(two_points["BCF_hz_estimate"]))
        self.assertAlmostEqual(two_points["VCL_px_s"], 20.0)

    def test_stationary_track_has_zero_speed_but_undefined_ratios(self):
        metrics = compute_track_metrics(
            [(frame, 2.0, 3.0) for frame in range(5)],
            fps=10.0,
        )

        self.assertEqual(metrics["VCL_px_s"], 0.0)
        self.assertTrue(math.isnan(metrics["LIN"]))
        self.assertTrue(math.isnan(metrics["STR"]))

    def test_bcf_estimate_counts_complete_lateral_cycles(self):
        fps = 20.0
        points = [
            (
                frame,
                float(frame),
                3.0 * math.sin(2.0 * math.pi * 2.0 * frame / fps),
            )
            for frame in range(121)
        ]

        metrics = compute_track_metrics(points, fps=fps)

        self.assertAlmostEqual(metrics["BCF_hz_estimate"], 2.0, delta=0.15)

    def test_rule_based_classifier_uses_speed_and_direction(self):
        progressive = compute_track_metrics(
            [(frame, frame * 2.0, 0.0) for frame in range(5)],
            fps=10.0,
        )
        non_progressive = compute_track_metrics(
            [(frame, float(frame % 2), 0.0) for frame in range(6)],
            fps=10.0,
        )
        immotile = compute_track_metrics(
            [(frame, 1.0, 1.0) for frame in range(5)],
            fps=10.0,
        )

        self.assertEqual(classify_motility(progressive), "Progressive")
        self.assertEqual(classify_motility(non_progressive), "Non-Progressive")
        self.assertEqual(classify_motility(immotile), "Immotile")

    def test_duplicate_detection_keeps_highest_confidence(self):
        detections = [
            self.make_detection(1, 0.7, (0.0, 0.0, 10.0, 10.0)),
            self.make_detection(2, 0.9, (0.2, 0.2, 10.2, 10.2)),
            self.make_detection(3, 0.8, (20.0, 20.0, 30.0, 30.0)),
        ]

        kept = remove_duplicate_detections(detections)

        self.assertEqual([item["track_id"] for item in kept], [2, 3])

    def test_invalid_fps_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_track_metrics([(0, 0.0, 0.0)], fps=0.0)

    @staticmethod
    def make_detection(track_id, confidence, box):
        return {
            "frame": 0,
            "time_s": 0.0,
            "track_id": track_id,
            "x": 0.0,
            "y": 0.0,
            "confidence": confidence,
            "box": box,
        }


if __name__ == "__main__":
    unittest.main()
