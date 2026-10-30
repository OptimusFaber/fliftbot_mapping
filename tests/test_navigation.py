"""Contract checks: calibrated replay, origin conversion and refusal of damaged maps."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from direct_udp_mc_localizer import load_map, transform
from lidar_profiles import load_profile
from map_bundle import validate_bundle
from marker98_pose import pose_in_marker98_frame
from robot_identity import mc_topic
from wheel_motion import telemetry_twist


class NavigationContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / 'deploy'
        self.bundle.mkdir()
        image = np.full((180, 210), 254, dtype=np.uint8)
        image[15, 15:190] = image[150, 15:190] = 0
        image[15:151, 15] = image[15:151, 189] = 0
        image[55:125, 90] = 0
        Image.fromarray(image).save(self.bundle / 'map.pgm')
        (self.bundle / 'map.yaml').write_text('image: map.pgm\nresolution: 0.05\norigin: [-3, -4, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
        # Unit fixture; actual PBStream loading is covered by the ROS integration run.
        (self.bundle / 'map.pbstream').write_bytes(b'unit-fixture-not-a-protobuf')
        self.reference = (1.2, -.7, math.radians(17))
        (self.bundle / 'origin_reference.json').write_text(json.dumps({
            'frame': 'cartographer_map', 'origin_map_xy_m': list(self.reference[:2]),
            'origin_map_yaw_rad': self.reference[2]}))
        config = {'schema': 'fliftbot.navigation_config.v1', 'points_frame': 'sensor',
                  'lidar_type': 'dephan_m360', 'max_range_m': 40,
                  'scan_to_base': {'x_m': 1.37, 'y_m': .315, 'yaw_deg': 31},
                  'odometry': {'speed_sign': -1, 'steering_sign': 1,
                               'steering_zero_deg': 0, 'wheelbase_m': 1.41, 'model': 'drive_wheel'}}
        (self.bundle / 'navigation_config.json').write_text(json.dumps(config))
        self.index()

    def index(self):
        files = [{'path': p.name, 'size': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in self.bundle.iterdir() if p.name != 'manifest.json']
        (self.bundle / 'manifest.json').write_text(json.dumps({'schema': 'fliftbot.deploy_catalog.v1', 'files': files}))

    def test_replay_calibration_and_nonzero_origin_yaw(self):
        points, *_ = load_map(self.bundle / 'map.yaml')
        body_pose = (1., -.3, .23)
        def inverse(pose):
            x, y, yaw = pose
            c, s = math.cos(yaw), math.sin(yaw)
            return -c*x-s*y, s*x-c*y, -yaw
        body_points = transform(points, inverse(body_pose))
        sensor_points = transform(body_points, inverse((1.37, .315, math.radians(31))))
        recording = self.root / 'recording'
        (recording / 'horizon').mkdir(parents=True)
        for i in range(3):
            np.savez(recording / 'horizon' / f'{i:04}.npz', xy=sensor_points,
                     stamp=np.asarray(120.+i*.04))  # frame field is optional canonical metadata
        out = self.root / 'localization'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/localize_recording.py'),
                                 '--recording', str(recording), '--map-dir', str(self.bundle),
                                 '--output', str(out), '--initial', '1.03', '-.32', '.24'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        with (out / 'trajectory.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        expected = pose_in_marker98_frame(*body_pose, *self.reference)
        np.testing.assert_allclose([float(rows[-1][key]) for key in ('x', 'y', 'alpha_rad')], expected, atol=.04)
        # ICP must not pass a scan with no usable returns.
        for path in (recording / 'horizon').glob('*.npz'):
            np.savez(path, xy=np.zeros((200, 2)), stamp=120.)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/localize_recording.py'),
                                 '--recording', str(recording), '--map-dir', str(self.bundle),
                                 '--output', str(self.root / 'reject')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)

    def test_corrupted_map_refused(self):
        validate_bundle(self.bundle)
        (self.bundle / 'map.pgm').write_bytes(b'bad map')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_bundle(self.bundle)

    def test_manifest_must_cover_reference_and_calibration(self):
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        manifest['files'] = [row for row in manifest['files'] if row['path'] != 'origin_reference.json']
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            validate_bundle(self.bundle)

    def test_origin_and_matrix_must_agree(self):
        np.save(self.bundle / 'pbstream_to_local_matrix.npy', np.eye(3))
        self.index()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            validate_bundle(self.bundle)

    def test_sensor_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_profile('pandar', self.bundle)

    def test_drive_wheel_pivot_and_reverse(self):
        v, wz = telemetry_twist(-.5, 90, speed_sign=-1, steering_zero_deg=0,
                                steering_sign=1, wheelbase=1.41, model='drive_wheel')
        self.assertAlmostEqual(v, 0)
        self.assertAlmostEqual(wz, .5/1.41)
        reverse = telemetry_twist(.5, 90, speed_sign=-1, steering_zero_deg=0,
                                  wheelbase=1.41, model='drive_wheel')
        self.assertAlmostEqual(reverse[1], -wz)
        self.assertEqual(mc_topic('flbot204'), '/fliftbot204/mc_telemetry')

    def test_continuous_nav_wire_and_angle_units(self):
        try:
            import zmq
        except ImportError:
            self.skipTest('install python3-zmq or use the navigation .venv for ContinuousNav transport test')
        from google.protobuf import descriptor_pool, message_factory
        from navigation_output import LidarSensorSender, LIDAR_VARIANCE
        from navigation_measurement import DESCRIPTOR
        # An independent consumer descriptor decodes the actual ZMQ payload.
        descriptor = descriptor_pool.DescriptorPool().AddSerializedFile(DESCRIPTOR.serialized_pb)
        consumer_type = descriptor.message_types_by_name["SensorMeasurement"]
        consumer = (message_factory.GetMessageClass(consumer_type) if hasattr(message_factory, "GetMessageClass")
                    else message_factory.MessageFactory().GetPrototype(consumer_type))
        context = zmq.Context()
        receiver = context.socket(zmq.SUB)
        receiver.setsockopt(zmq.SUBSCRIBE, b'')
        port = receiver.bind_to_random_port('tcp://127.0.0.1')
        sender = LidarSensorSender(f'tcp://127.0.0.1:{port}')
        try:
            deadline = time.monotonic() + 3
            payload = None
            while time.monotonic() < deadline:
                sender.send(1793369600., 2.5, -1.2, 90.)
                if receiver.poll(50):
                    payload = receiver.recv()
                    break
            self.assertIsNotNone(payload, 'ZMQ pose transport did not deliver')
            measurement = consumer.FromString(payload)
            self.assertEqual(measurement.sensor_type, 0)  # ContinuousNav top_navigation
            self.assertEqual(measurement.timestamp, 1793369600.)
            np.testing.assert_allclose(measurement.value, [2.5, -1.2, math.pi/2])
            np.testing.assert_allclose(measurement.variance, LIDAR_VARIANCE)
        finally:
            sender.close()
            receiver.close(linger=0)
            context.term()

    def test_bundled_profiles_and_maps(self):
        for sensor in ('dephan', 'pandar'):
            profile = load_profile(sensor)
            self.assertTrue(profile['pbstream'].is_file())
            self.assertTrue(profile['reference'].is_file())
            load_map(profile['map'])


if __name__ == '__main__':
    unittest.main()
