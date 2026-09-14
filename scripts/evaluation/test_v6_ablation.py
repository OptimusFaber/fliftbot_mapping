import json
import tempfile
import unittest
from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit", HERE / "audit_v6_ablation.py")
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


class V6AblationTests(unittest.TestCase):
    def test_ray_convention_is_explicit(self):
        result = audit.ray_convention(
            Path("ros-data/recordings/real/20260908/160617_dephan_m360_room/observations.json"),
            Path("ros-data/recordings/real/20260908/160617_dephan_m360_room/camera.json"),
        )
        self.assertGreater(result["direct_rebuilt_vs_stored"]["median_deg"], 20.0)
        self.assertLess(result["stored_vs_rebuilt_rotated_plus90deg_z"]["median_deg"], 1.0)

    def test_provenance_flags_historical_enc_unix(self):
        result = audit.provenance(
            Path("ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate"),
            Path("fliftbot_mapping/standalone_cartographer/runner.cc"),
            Path("fliftbot_mapping/scripts/evaluation/check_standalone_inputs.py"),
        )
        self.assertFalse(result["strict_enc_t_only"])
        self.assertIn("enc_unix", result["cartographer_runner_encoder_fields"])


if __name__ == "__main__":
    unittest.main()
