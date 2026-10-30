"""ROS subprocess shutdown without repeated SIGINT interrupting cleanup."""
import signal
import rclpy
from rclpy.executors import ExternalShutdownException


def spin_node(node, before_destroy=None):
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        # Keep shutdown out of signal handlers: interrupting an rclpy context
        # lock (notably with a paused simulation clock) can deadlock cleanup.
        while not stopping and node.context.ok():
            rclpy.spin_once(node, timeout_sec=.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if before_destroy:
            before_destroy()
        node.destroy_node()
        rclpy.try_shutdown()
