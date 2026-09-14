import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import nav_marker_map_notebook_style as m


class NotebookNavTests(unittest.TestCase):
    def test_nav_parser_uses_only_nav_columns_and_utc(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "move.log"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["t", "x", "y", "dgr", "nav_t", "nav_x", "nav_y", "nav_dgr"])
                w.writeheader()
                w.writerow({"t": "bad", "x": "99", "y": "99", "dgr": "99", "nav_t": "2026_09_08_13_00_00_000", "nav_x": "1", "nav_y": "2", "nav_dgr": "90"})
                w.writerow({"t": "bad", "x": "99", "y": "99", "dgr": "99", "nav_t": "2026_09_08_13_00_01_000", "nav_x": "2", "nav_y": "3", "nav_dgr": "180"})
            t, p, meta = m.load_nav(path)
            self.assertEqual(meta["fields"], ["nav_t", "nav_x", "nav_y", "nav_dgr"])
            self.assertEqual(meta["forbidden_fields"], ["t", "x", "y", "dgr"])
            self.assertAlmostEqual(t[1] - t[0], 1.0)
            np.testing.assert_allclose(p[0], [1, 2, np.deg2rad(90)])

    def test_notebook_ray_plane_formula(self):
        payload = {"data": [{"1": {"code": 98, "ray": [0, 0, 1]}}]}
        rows = m.raw_rows(payload, np.asarray([10.0]), np.asarray([0.0, 10.0]), np.asarray([[2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]), np.asarray([1.0, 2.0, 1.0]), 3.0, 0.0)
        np.testing.assert_allclose(rows[0]["point_xy_m"], [3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
