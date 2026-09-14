#!/usr/bin/env python3
"""Synthetic regression tests for the experimental V7 geometry contract."""
from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ceiling_marker_localization_v7 as v7


def make_rows(height: float, camera_offset: np.ndarray, targets: dict[int, np.ndarray], count: int = 8, noise: float = 0.0):
    rows = []
    # Poses are deliberately rotated and translated so this exercises the
    # rotation-axis/body frame and not only an origin-at-zero special case.
    for frame in range(count):
        yaw = math.radians(-35.0 + frame * 17.0)
        pose = np.asarray([0.4 + 0.03 * frame, -0.2 + 0.02 * frame, yaw])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.asarray([[c, -s], [s, c]])
        origin = np.asarray([*(pose[:2] + R @ camera_offset[:2]), camera_offset[2]])
        for code, target in targets.items():
            direction_map = np.asarray([*(target[:2] - origin[:2]), height - origin[2]])
            direction_map /= np.linalg.norm(direction_map)
            ray = np.asarray([*(R.T @ direction_map[:2]), direction_map[2]])
            if noise:
                ray += np.asarray([noise * math.sin(frame + code), noise * math.cos(frame), 0.0])
                ray /= np.linalg.norm(ray)
            rows.append({
                "frame_index": frame,
                "timestamp_s": 100.0 + frame,
                "track_id": f"{code}-{frame}",
                "marker_id": code,
                "ray_optical_unit": ray.tolist(),
            })
    return rows


class V7GeometryTests(unittest.TestCase):
    def test_height_and_mounts(self):
        times = np.arange(100.0, 108.0)
        poses = np.asarray([[0.4 + 0.03 * i, -0.2 + 0.02 * i, math.radians(-35 + 17 * i)] for i in range(8)])
        for height, mount in ((3.03, np.asarray([1.125, -0.28, 1.27])), (3.13, np.asarray([0.4, 0.15, 1.6])), (4.2, np.asarray([-0.2, 0.3, 1.1]))):
            targets = {10: np.asarray([2.0, 1.0, height]), 20: np.asarray([1.0, 2.2, height]), 30: np.asarray([3.0, 2.0, height])}
            projected = v7.project_rows(make_rows(height, mount, targets), times, poses, mount, 0.0)
            fits = []
            for code in targets:
                fit = v7.triangulate([r for r in projected if r["marker_id"] == code], 0.03, 80)
                self.assertIsNotNone(fit)
                fits.append(fit)
            estimate = v7.estimate_height(fits, mount[2], 0.5, 8.0)
            self.assertIsNotNone(estimate)
            self.assertAlmostEqual(estimate["ceiling_z_m"], height, delta=0.01)

    def test_lidar_pose_camera_lever_arm_and_timestamp_offset(self):
        height = 3.13
        axis_camera = np.asarray([1.125, -0.28, 1.27])
        lidar = np.asarray([1.29, -0.32, 2.03])
        camera_from_lidar = axis_camera.copy(); camera_from_lidar[:2] -= lidar[:2]
        times = np.arange(100.0, 108.0)
        poses = np.asarray([[0.4 + 0.03 * i, -0.2 + 0.02 * i, math.radians(-35 + 17 * i)] for i in range(8)])
        targets = {10: np.asarray([2.0, 1.0, height]), 20: np.asarray([1.0, 2.2, height]), 30: np.asarray([3.0, 2.0, height])}
        # A trajectory timestamp delayed by 20 ms is corrected by time_offset.
        rows = make_rows(height, camera_from_lidar, targets)
        for row in rows: row["timestamp_s"] -= 0.02
        projected = v7.project_rows(rows, times, poses, camera_from_lidar, 0.02)
        self.assertEqual(len(projected), len(rows))
        fit = v7.triangulate([r for r in projected if r["marker_id"] == 10], 0.03, 80)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["point_xyz_m"][2], height, delta=0.01)

    def test_physical_duplicate_decoder_ids_are_collapsed(self):
        height = 3.03; mount = np.asarray([1.125, -0.28, 1.27])
        targets = {10: np.asarray([2.0, 1.0, height]), 11: np.asarray([2.01, 1.01, height]), 20: np.asarray([1.0, 2.2, height])}
        times = np.arange(100.0, 108.0)
        poses = np.asarray([[0.4 + 0.03 * i, -0.2 + 0.02 * i, math.radians(-35 + 17 * i)] for i in range(8)])
        projected = v7.project_rows(make_rows(height, mount, targets, noise=0.0002), times, poses, mount, 0.0)
        markers = v7.aggregate(projected, height, 4, 0.20)
        self.assertEqual(len(markers), 2)
        self.assertTrue(any(m["identity_ambiguous"] for m in markers))

    def test_height_abstains_without_support(self):
        self.assertIsNone(v7.estimate_height([], 1.2, 0.5, 8.0))

    def test_forbidden_pose_fields_are_ignored(self):
        base = [
            {"enc_t": "2026_09_08_13_00_00_000000", "enc_v": 0.2, "enc_dgr": 5.0},
            {"enc_t": "2026_09_08_13_00_01_000000", "enc_v": 0.2, "enc_dgr": 5.0},
        ]
        polluted = [
            {**row, "x": 999.0, "y": -999.0, "dgr": 123.0, "nav_x": 77.0, "nav_y": 88.0, "nav_dgr": 66.0}
            for row in base
        ]
        with tempfile.TemporaryDirectory() as directory:
            clean_path = Path(directory) / "clean.jsonl"
            polluted_path = Path(directory) / "polluted.jsonl"
            clean_path.write_text("\n".join(json.dumps(row) for row in base) + "\n")
            polluted_path.write_text("\n".join(json.dumps(row) for row in polluted) + "\n")
            clean = v7.load_encoder_log(clean_path, 1.410, -1.0, 1.0, 0.0, 0.0)
            changed = v7.load_encoder_log(polluted_path, 1.410, -1.0, 1.0, 0.0, 0.0)
        np.testing.assert_allclose(clean[0], changed[0])
        np.testing.assert_allclose(clean[1], changed[1])
        self.assertEqual(changed[2]["source_fields"], ["enc_t", "enc_v", "enc_dgr"])
        self.assertIn("nav_x", changed[2]["forbidden_fields"])


if __name__ == "__main__":
    unittest.main()
