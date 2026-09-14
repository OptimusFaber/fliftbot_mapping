import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_cartographer_nav_markers as m


class CompareCartographerNavTest(unittest.TestCase):
    def test_nav_timestamp_is_utc_epoch(self):
        self.assertAlmostEqual(m.stamp("2026_09_08_13_06_17_172679"), 1788872777.172679, places=5)

    def test_lidar_to_axis_offset_sign(self):
        # lidar pose at (10, 4), yaw=90deg; axis = lidar - R*[1.29,-.32]
        p = np.asarray([[10.0, 4.0, math.pi / 2]])
        c0, s0 = np.cos(p[:, 2]), np.sin(p[:, 2])
        R = np.stack([np.stack([c0, -s0], 1), np.stack([s0, c0], 1)], 1)
        axis = p[:, :2] - np.einsum("nij,j->ni", R, np.asarray([1.29, -.32]))
        np.testing.assert_allclose(axis[0], [9.68, 2.71], atol=1e-9)

    def test_initial_pose_alignment_does_not_fit_full_trajectory(self):
        nav = [{"timestamp_s": 0.0, "x": 1.0, "y": 2.0, "yaw": 0.0},
               {"timestamp_s": 1.0, "x": 2.0, "y": 2.0, "yaw": 0.0}]
        cart = np.asarray([[5.0, 6.0, math.pi / 2], [5.0, 7.0, math.pi / 2]])
        with tempfile.TemporaryDirectory() as d:
            map_dir = Path(d); (map_dir / "map.yaml").write_text("resolution: 1.0\norigin: [0.0, 0.0, 0.0]\n")
            from PIL import Image
            Image.new("L", (20, 20), 255).save(map_dir / "map.pgm")
            report = m.trajectory_report(np.asarray([0.0, 1.0]), cart, nav, np.zeros(2), map_dir, map_dir)
        self.assertAlmostEqual(report["axis_frame_initial_pose_alignment"]["rotation_deg"], 90.0, places=6)
        self.assertEqual(report["samples"], 5)


if __name__ == "__main__":
    unittest.main()
