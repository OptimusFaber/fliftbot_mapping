"""ROS-independent ContinuousNav pose sender (metres, radians, Unix seconds)."""
import math
from navigation_measurement import SensorMeasurement

LIDAR_VARIANCE = [0.05 ** 2, 0.05 ** 2, math.radians(3.0) ** 2]


class LidarSensorSender:
    """Send lidar pose as a full-pose ContinuousNav measurement."""

    def __init__(self, socket_address: str) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzmq is required for ContinuousNav output") from exc
        self.zmq = zmq
        self.message_type = SensorMeasurement
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        # Never build a seconds-long pose backlog.  A localization consumer
        # needs the newest pose; replaying stale poses destabilizes its UKF.
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.connect(socket_address)
        self.sent_count = 0

    def send(self, timestamp: float, x: float, y: float,
             alpha_deg: float) -> None:
        # ContinuousNav's top_navigation measurement is [x_m, y_m, yaw_rad].
        # The ROS/CSV interface remains in degrees for operators.
        message = self.message_type(
            sensor_type=self.message_type.SensorType.Value("top_navigation"),
            timestamp=timestamp,
            value=[x, y, math.radians(alpha_deg)],
            variance=LIDAR_VARIANCE,
        )
        self.socket.send(message.SerializeToString(), flags=self.zmq.NOBLOCK)
        self.sent_count += 1

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()

