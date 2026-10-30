"""Shared live/offline encoder convention; steering is measured at the drive wheel."""
import math


def telemetry_twist(speed_mps, wheel_angle_deg, *, speed_sign, steering_zero_deg,
                    wheelbase, model, steering_sign=1.0):
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    speed = speed_sign * speed_mps
    delta = steering_sign * math.radians(wheel_angle_deg - steering_zero_deg)
    if model == "drive_wheel":
        return speed * math.cos(delta), speed * math.sin(delta) / wheelbase
    if model == "bicycle":
        return speed, speed * math.tan(delta) / wheelbase
    if model == "legacy_sin":
        return speed, speed * math.sin(delta) / wheelbase
    raise ValueError(f"unsupported odometry model {model!r}")
