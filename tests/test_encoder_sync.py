import math

from flifltbot.mapping.encoder_sync import (
    motion_from_encoders, samples_have_usable_yaw_rate,
    set_wheel_direction_offset,
)
from flifltbot.mapping.trajectory_graph import (
    _sequential_scan_agrees_with_wheel,
)


def test_gentle_yaw_rate_is_not_discarded():
    samples = [{'enc_wz': 0.032}] * 90 + [{'enc_wz': 0.0}] * 10
    assert samples_have_usable_yaw_rate(samples)


def test_missing_yaw_rate_stays_disabled():
    samples = [{'enc_wz': 0.0}] * 100
    assert not samples_have_usable_yaw_rate(samples)


def test_tricycle_uses_wheel_speed_projection_and_computed_yaw():
    set_wheel_direction_offset(0.0)
    a = {'enc_t': 0.0, 'enc_v': 1.0, 'enc_dgr': 60.0}
    b = {'enc_t': 0.1, 'enc_v': 1.0, 'enc_dgr': 60.0}
    dx, _dy, dtheta = motion_from_encoders(a, b)
    assert math.isclose(dx, -0.05, rel_tol=0.01)
    assert math.isclose(
        dtheta, -math.sin(math.radians(60.0)) / 1.41 * 0.1,
        rel_tol=1e-6,
    )


def test_reverse_changes_yaw_sign_for_same_steering_angle():
    set_wheel_direction_offset(0.0)
    a = {'enc_t': 0.0, 'enc_v': -1.0, 'enc_dgr': 60.0}
    b = {'enc_t': 0.1, 'enc_v': -1.0, 'enc_dgr': 60.0}
    dx, _dy, dtheta = motion_from_encoders(a, b)
    assert math.isclose(dx, 0.05, rel_tol=0.01)
    assert math.isclose(
        dtheta, math.sin(math.radians(60.0)) / 1.41 * 0.1,
        rel_tol=1e-6,
    )


def test_normal_wheel_motion_is_not_discarded_at_20hz():
    """A 15 mm encoder step is motion, despite being below ICP STATIC_XY."""
    set_wheel_direction_offset(0.0)
    a = {'enc_t': 0.00, 'enc_v': -0.295, 'enc_dgr': 0.0}
    b = {'enc_t': 0.05, 'enc_v': -0.295, 'enc_dgr': 0.0}

    dx, dy, dtheta = motion_from_encoders(a, b)

    assert math.isclose(dx, 0.295 * 0.05, rel_tol=1e-6)
    assert abs(dy) < 1e-12
    assert abs(dtheta) < 1e-12


def test_sequential_icp_cannot_erase_forward_wheel_motion():
    wheel = [0.18, 0.0, 0.0]
    assert not _sequential_scan_agrees_with_wheel([0.0, 0.0, 0.0], wheel)
    assert _sequential_scan_agrees_with_wheel([0.16, 0.02, 0.01], wheel)


def test_wheel_motion_ignores_injected_global_gt_fields():
    set_wheel_direction_offset(0.0)
    plain_a = {'enc_t': 0.0, 'enc_v': -0.30, 'enc_dgr': 10.0}
    plain_b = {'enc_t': 0.05, 'enc_v': -0.30, 'enc_dgr': 10.0}
    injected_a = dict(
        plain_a, odom_x=1000.0, odom_y=-500.0, odom_yaw=2.4, enc_wz=9.0
    )
    injected_b = dict(
        plain_b, odom_x=-1000.0, odom_y=500.0, odom_yaw=-2.4, enc_wz=-9.0
    )

    assert motion_from_encoders(plain_a, plain_b) == motion_from_encoders(
        injected_a, injected_b
    )
