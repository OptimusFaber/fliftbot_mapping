"""Navigation output axes, including the X-up/Y-right reflected frame."""
import json
import math
from pathlib import Path
import sys
import tempfile
import csv
import hashlib
import shutil
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from marker98_pose import pose_in_marker98_frame
from map_bundle import read_reference, read_y_axis_sign, validate_bundle
from trajectory_marker98_export import normalize


class NavigationAxes(unittest.TestCase):
    def test_trajectory_export_accepts_reflected_origin_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"origin_map_xy_m": [10., 20.],
                                             "origin_map_yaw_rad": math.pi / 2,
                                             "y_axis_sign": -1}))
            source = root / "trajectory.json"
            source.write_text(json.dumps({"poses": [{"x": 11., "y": 20.,
                "qw": 1., "qz": 0., "timestamp_universal": 621355968000000000}]}))
            output = root / "trajectory.csv"
            normalize(source, output, reference)
            with output.open() as stream:
                row = next(csv.DictReader(stream))
            np.testing.assert_allclose([float(row[k]) for k in ("x", "y", "alpha")],
                                       [0., 1., math.pi / 2], atol=1e-12)

    def test_validator_rejects_web_origin_drift_even_with_valid_checksum(self):
        bundle = ROOT / "maps/tver_agro_20261109"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for source in bundle.iterdir():
                if source.is_file():
                    shutil.copy2(source, target / source.name)
            metadata_path = target / "map_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata_path.unlink()
            metadata["origin_pixel_xy_tver"] = [253, 4203]
            metadata_path.write_text(json.dumps(metadata))
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest_path.unlink()
            for item in manifest["files"]:
                if item["path"] == metadata_path.name:
                    item.update(size=metadata_path.stat().st_size,
                                sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "pixel origin"):
                validate_bundle(target)

    def test_up_right_and_heading(self):
        # Conventional source axes: X right, Y up. Origin (10,20).
        reference = (10., 20., math.pi / 2)
        cases = [((10., 20., math.pi / 2), (0., 0., 0.)),
                 ((10., 21., math.pi / 2), (1., 0., 0.)),
                 ((11., 20., 0.), (0., 1., math.pi / 2)),
                 ((9., 20., math.pi), (0., -1., -math.pi / 2))]
        for pose, expected in cases:
            with self.subTest(pose=pose):
                np.testing.assert_allclose(
                    pose_in_marker98_frame(*pose, *reference, y_axis_sign=-1),
                    expected, atol=1e-12)

    def test_legacy_frame_keeps_right_handed_axes(self):
        np.testing.assert_allclose(
            pose_in_marker98_frame(11., 20., 0., 10., 20., math.pi / 2),
            (0., -1., -math.pi / 2), atol=1e-12)

    def test_tver_bundle_matches_image_axes_and_matrices(self):
        bundle = ROOT / "maps/tver_agro_20261109"
        ref = read_reference(bundle / "origin_reference.json")
        sign = read_y_axis_sign(bundle / "origin_reference.json")
        self.assertEqual(sign, 1)
        geometry = json.loads((bundle / "geometry_verification.json").read_text())
        pixel_to_pb = np.array(geometry["base_pixel_to_pbstream_map_metres"])
        transform = np.load(bundle / "pbstream_to_local_matrix.npy")
        web = np.load(bundle / "map_transform_matrix.npy")
        for pixel, expected, expected_web in [
                ((152, 4103), (0, 0), (252, 4203)),
                ((152, 4003), (0, 1), (252, 4103)),
                ((252, 4103), (1, 0), (352, 4203))]:
            point = pixel_to_pb @ [*pixel, 1.]
            actual = pose_in_marker98_frame(*point[:2], ref[2], *ref,
                                           y_axis_sign=sign)
            np.testing.assert_allclose(actual[:2], expected, atol=1e-6)
            np.testing.assert_allclose((transform @ point)[:2], expected, atol=1e-6)
            np.testing.assert_allclose((web @ [*expected, 1])[:2], expected_web)


if __name__ == "__main__":
    unittest.main()
