import tempfile
import unittest
from pathlib import Path

import numpy as np

import interactive_map_measure as m


class InteractiveMapMeasureTest(unittest.TestCase):
    def test_yaml_and_relative_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "room.pgm"
            image.write_bytes(b"not-read-by-parser")
            yaml = root / "map.yaml"
            yaml.write_text("image: room.pgm\nresolution: 0.05\norigin: [-1.0, 2.0, 1.57]\n")
            image_path, resolution, origin = m.resolve_map_input(yaml)
            self.assertEqual(image_path, image.resolve())
            self.assertEqual(resolution, 0.05)
            self.assertEqual(origin, (-1.0, 2.0))

    def test_ros_row_zero_is_top(self):
        np.testing.assert_allclose(
            m.pixel_to_world([0, 0], 100, 0.1, (2.0, -3.0)), [2.05, 6.95]
        )
        self.assertAlmostEqual(m.world_distance_cm([0, 0], [3, 4], 100, 0.1, (0, 0)), 50.0)

    def test_rotation_round_trip_and_invariant_length(self):
        point = np.array([12.25, 4.75])
        for angle in (0, 90, 180, 270, 33):
            shown = m.rotate_display_point(point, (40, 20), angle)
            recovered = m.inverse_rotate_display_point(shown, (40, 20), angle)
            np.testing.assert_allclose(recovered, point, atol=1e-9)
        before = m.world_distance_cm([2, 3], [18, 10], 20, 0.05, (4, 5))
        after = m.world_distance_cm([2, 3], [18, 10], 20, 0.05, (99, -8))
        self.assertAlmostEqual(before, after, places=10)


if __name__ == "__main__":
    unittest.main()
