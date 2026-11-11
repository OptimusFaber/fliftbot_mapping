"""ROS-independent ContinuousNav pose sender (metres, radians, Unix seconds)."""
import math
from navigation_measurement import SensorMeasurement

LIDAR_VARIANCE = [0.05 ** 2, 0.05 ** 2, math.radians(3.0) ** 2]


def send_sensor_measurement(socket, timestamp: float, value, variance=(),
                            sensor_type="top_navigation") -> None:
    """Send one SensorMeasurement protobuf over an existing ZMQ PUB socket.

    The receiver binds a SUB socket and subscribes to b"" (no topic prefix).
    Reuse the connected socket; PUB/SUB discovery must complete before sending.
    timestamp: Unix seconds. Pose value: [x_m, y_m, yaw_rad].
    sensor_type: enum name, e.g. "top_navigation" or "lidar".
    Raises zmq.Again if the nonblocking send would block.
    """
    import zmq
    message = SensorMeasurement(
        sensor_type=SensorMeasurement.SensorType.Value(sensor_type),
        timestamp=timestamp,
        value=value,
        variance=variance,
    )
    socket.send(message.SerializeToString(), flags=zmq.NOBLOCK)


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
        send_sensor_measurement(
            self.socket, timestamp, [x, y, math.radians(alpha_deg)], LIDAR_VARIANCE,
            sensor_type="top_navigation",
        )
        self.sent_count += 1

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()

