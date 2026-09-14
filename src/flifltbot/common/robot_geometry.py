"""Canonical measured robot geometry shared by control, logging and mapping."""

from __future__ import annotations

DRIVE_WHEEL_X_M = 1.410
# Physical steering/drive wheel diameter.  enc_v is already linear m/s, so
# this value is metadata and must not be applied a second time.
DRIVE_WHEEL_DIAMETER_M = 0.230
# Hardware/src convention: positive encoder speed moves the rear-axle origin
# along body -X.  Steering is already physical: 0 straight, + left.
ENCODER_SPEED_TO_BODY_SIGN = -1.0
ENCODER_STEERING_SIGN = 1.0
ENCODER_STEERING_OFFSET_RAD = 0.0
ROBOT_ACCELERATION_MPS2 = 2.0
ROBOT_BRAKE_DECELERATION_MPS2 = 2.3
ROBOT_STEERING_RATE_DEG_S = 60.0
UP_LIDAR_X_FORKS_DOWN_M = 1.015
UP_LIDAR_X_FORKS_UP_M = 0.985
NOMINAL_FORK_LIFT_M = 0.15


def up_lidar_x(fork_lift_m=0.0):
    """Interpolate the measured up-LiDAR lever arm over the nominal lift."""
    lift = max(0.0, min(NOMINAL_FORK_LIFT_M, float(fork_lift_m)))
    ratio = lift / NOMINAL_FORK_LIFT_M
    return (
        UP_LIDAR_X_FORKS_DOWN_M
        + ratio * (UP_LIDAR_X_FORKS_UP_M - UP_LIDAR_X_FORKS_DOWN_M)
    )
