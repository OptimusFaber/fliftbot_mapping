"""ROS TF -> pose exporter -> real local ZMQ transport, with no robot connection."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import rclpy
    import zmq
    from geometry_msgs.msg import TransformStamped
    from live_marker98_pose import Marker98PoseNode
    from navigation_measurement import SensorMeasurement
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'requires ROS and pyzmq')
class LivePoseRuntime(unittest.TestCase):
    def test_navigation_enabled_start_and_offset_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reference=root/'reference.json'
            reference.write_text('{"origin_map_xy_m": [0,0], "origin_map_yaw_rad": 0}')
            ctx=zmq.Context();receiver=ctx.socket(zmq.SUB);receiver.setsockopt(zmq.SUBSCRIBE,b'')
            port=receiver.bind_to_random_port('tcp://127.0.0.1')
            args=argparse.Namespace(marker_reference=str(reference),scenario_config='',journal=str(root/'run.jsonl'),
                csv=str(root/'poses.csv'),map_frame='audit_map',base_frame='audit_base',
                pose_offset_x_m=-.04,pose_offset_y_m=-.045,topic='/audit/pose',disable_sensor_sender=False,
                send_to_navigation=True,sensor_socket=f'tcp://127.0.0.1:{port}',rate=10.,sensor_send_rate=10.,
                max_sensor_age=.5,scan_topic='/audit/scan',odom_topic='/audit/odom')
            node=None
            rclpy.init()
            try:
                node=Marker98PoseNode(args)  # Includes startup variance message.
                message=None;deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    tf=TransformStamped();tf.header.frame_id=args.map_frame;tf.child_frame_id=args.base_frame
                    tf.header.stamp=node.get_clock().now().to_msg()
                    tf.transform.translation.x=2.;tf.transform.translation.y=3.
                    tf.transform.rotation.z=math.sin(math.pi/4);tf.transform.rotation.w=math.cos(math.pi/4)
                    node.buffer.set_transform(tf,'test');node.tick()
                    if receiver.poll(30):
                        message=SensorMeasurement.FromString(receiver.recv());break
                self.assertIsNotNone(message)
                self.assertAlmostEqual(message.value[0],2.045)
                self.assertAlmostEqual(message.value[1],2.96)
                self.assertAlmostEqual(message.value[2],math.pi/2)
                count=node.sensor_sender.sent_count
                # Reusing the same TF must not result in a duplicate send.
                node.next_sensor_send_monotonic=0.;node.tick()
                self.assertEqual(node.sensor_sender.sent_count,count)
            finally:
                if node: node.destroy_node()
                rclpy.try_shutdown();receiver.close(linger=0);ctx.term()
            rows=[json.loads(line) for line in (root/'run.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['kind'],'run_end')


if __name__=='__main__': unittest.main()
