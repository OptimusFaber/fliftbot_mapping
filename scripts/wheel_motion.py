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


def resolve_odometry_model(model, use_legacy_sin, fallback="drive_wheel"):
    """Keep the historical sin/tan launch switch compatible with explicit models."""
    legacy = use_legacy_sin.strip().lower()
    alias_model = None
    if legacy:
        if legacy not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            raise ValueError("use_legacy_sin must be true or false, or empty")
        alias_model = "legacy_sin" if legacy in ("1", "true", "yes", "on") else "bicycle"
    if model and alias_model and model != alias_model:
        raise ValueError("odom_model conflicts with use_legacy_sin")
    resolved = model or alias_model or fallback
    if resolved not in ("drive_wheel", "legacy_sin", "bicycle"):
        raise ValueError(f"unsupported odometry model {resolved!r}")
    return resolved
