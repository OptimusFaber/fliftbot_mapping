from flifltbot.mapping.pipeline import (
    FORBIDDEN_MAPPING_MOTION_FIELDS,
    sanitize_mapping_encoder_sample,
)


def test_mapping_strips_gt_pose_and_body_yaw_rate():
    sample = {
        'enc_t': 1.0,
        'enc_v': 0.7,
        'enc_dgr': 12.0,
        'enc_wz': 0.9,
        'odom_x': 10.0,
        'odom_y': 20.0,
        'odom_yaw': 1.2,
    }

    sanitized = sanitize_mapping_encoder_sample(sample)

    assert sanitized['enc_v'] == 0.7
    assert sanitized['enc_dgr'] == 12.0
    assert not (FORBIDDEN_MAPPING_MOTION_FIELDS & sanitized.keys())


def test_mapping_preserves_missing_encoder_record():
    assert sanitize_mapping_encoder_sample(None) is None
