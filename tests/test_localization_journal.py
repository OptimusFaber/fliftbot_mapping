import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from localization_journal import Journal, load_scenario
from localization_report import evaluate, interpolate


class JournalTests(unittest.TestCase):
    def run_case(self, values, reference=None, **overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run.jsonl"
            cfg = {**load_scenario(), "orientation_error_deg": 2, **overrides}
            journal = Journal(path, cfg)
            for now, stamp, pose in values:
                journal.observe(now, stamp, pose)
            journal.close(values[-1][0])
            return evaluate(path, reference), [json.loads(v) for v in path.read_text().splitlines()]

    def test_good_trajectory_and_wrapped_angle(self):
        values = [(i/10, i/10, (0, 0, math.radians(179))) for i in range(21)]
        refs = [(i/10, 0, 0, math.radians(-180)) for i in range(21)]
        report, _ = self.run_case(values, refs)
        self.assertEqual(report["classification"], "successful_single_run")
        self.assertAlmostEqual(report["orientation_error_deg"]["p95"], 1)
        self.assertEqual(report["failure_count"], 0)

    def test_stale_duplicate_is_not_available_forever(self):
        report, records = self.run_case([(i/10, 0., (0, 0, 0)) for i in range(21)])
        self.assertAlmostEqual(report["availability"], .25)
        self.assertEqual(report["loss_count"], 1)
        self.assertAlmostEqual(report["loss_duration_s"], 1.5)
        samples = [r for r in records if r["kind"] == "sample"]
        self.assertFalse(samples[1]["new_sample"])
        self.assertEqual(report["classification"], "insufficient_evidence")

    def test_missing_period_between_samples(self):
        report, _ = self.run_case([(0., 0., (0, 0, 0)), (2., 2., (0, 0, 0))])
        self.assertAlmostEqual(report["availability"], .25)
        self.assertEqual(report["loss_count"], 1)

    def test_threshold_episode_has_value_and_scenario(self):
        values = [(i/10, i/10, (.1 if 5 <= i < 15 else 0, 0, 0)) for i in range(21)]
        refs = [(i/10, 0, 0, 0) for i in range(21)]
        report, _ = self.run_case(values, refs)
        self.assertEqual(report["failure_count"], 1)
        self.assertEqual(len(report["error_events"]), 1)
        self.assertEqual(report["error_events"][0]["threshold"], .05)
        self.assertAlmostEqual(report["error_events"][0]["peak_error"], .1)
        self.assertEqual(report["classification"], "partially_successful")

    def test_reference_gap_prevents_false_success(self):
        values = [(i/10, i/10, (0, 0, 0)) for i in range(21)]
        report, _ = self.run_case(values, [(0, 0, 0, 0), (2, 0, 0, 0)])
        self.assertEqual(report["classification"], "insufficient_evidence")
        self.assertIsNone(interpolate([(0,0,0,0), (2,0,0,0)], [0,2], 1, .2))

    def test_missing_angle_threshold_prevents_success(self):
        values = [(i/10, i/10, (0,0,0)) for i in range(21)]
        report, _ = self.run_case(values, [(i/10,0,0,0) for i in range(21)], orientation_error_deg=None)
        self.assertEqual(report["classification"], "insufficient_evidence")

    def test_clock_reset_rejected(self):
        with self.assertRaisesRegex(ValueError, "clock moved backwards"):
            self.run_case([(2.,2.,(0,0,0)),(1.,1.,(0,0,0))])

    def test_short_exceedance_is_not_a_failure_episode(self):
        values = [(i/10, i/10, (.1 if i == 5 else 0, 0, 0)) for i in range(21)]
        report, _ = self.run_case(values, [(i/10,0,0,0) for i in range(21)])
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["error_events"], [])
        self.assertLess(report["position_within_threshold_fraction"], 1)

    def test_reference_angle_interpolation_short_path(self):
        angle = interpolate([(0,0,0,math.radians(179)), (.1,0,0,math.radians(-179))], [0,.1], .05, .2)[2]
        self.assertAlmostEqual(abs(angle), math.pi)

    def test_journal_refuses_overwrite_and_handles_bad_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run.jsonl"
            journal = Journal(path, load_scenario())
            journal.observe(1., 1., (0,0,0), odom={"speed": float("nan")})
            journal.close(1.)
            records = [json.loads(v) for v in path.read_text().splitlines()]
            self.assertIsNone([r for r in records if r["kind"] == "sample"][0]["odom"]["speed"])
            with self.assertRaises(FileExistsError):
                Journal(path, load_scenario())

    def test_invalid_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"cfg.json"
            path.write_text('{"max_pose_age_s": -1}')
            with self.assertRaises(ValueError):
                load_scenario(path)


if __name__ == "__main__":
    unittest.main()
