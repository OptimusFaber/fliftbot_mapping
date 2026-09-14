"""Shared mapping constants and presets."""

from __future__ import annotations

import math
from pathlib import Path

import requests

from flifltbot.common.robot_geometry import (
    DRIVE_WHEEL_X_M, UP_LIDAR_X_FORKS_DOWN_M, UP_LIDAR_X_FORKS_UP_M,
)

BASE_CANDIDATES = (
    'http://10.0.2.2:10101',
    'http://10.0.2.1:10101',
    'http://10.0.2.3:10101',
    'http://10.0.2.4:10101',
    'http://10.0.2.5:10101',
    'http://10.0.2.6:10101',
    'http://10.0.2.7:10101',
)
BASE = BASE_CANDIDATES[0]
HZ = 10.0
ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / 'live_map_result.json'
OUT_PNG = ROOT / 'live_map.png'
ENCODER_LOG_NAME = 'encoder.jsonl'

# Cameras available on /raw_points. Edit MAP_CAMERAS to choose which ones
# are used when building the global map (any non-empty subset).
AVAILABLE_CAMERAS = ('left', 'right', 'up')
# ``up`` is the canonical name everywhere.  ``top`` is accepted only when
# reading old recordings; new metadata and output always use ``up``.
CAMERA_ALIASES = {'top': 'up'}
# Robot frame: origin = rotation center (midpoint between rear fork wheels).
# +X forward (head/lidar direction), +Y left. Matches models/flifltbot/model.sdf.
LOW_LIDAR_X = 1.71          # m, dock lidar center from rear ICR
LOW_LIDAR_HALF_Y = 0.385    # m, corrected measured mount offset
LOW_LIDAR_YAW = math.pi / 4.0  # rad, floor lidars mounted ±45° (sim SDF)
LOW_LIDAR_Z = 0.135         # m, cylinder center; stand bottom at 0.10 on corpus
# Upper-LiDAR lever arms from the rear-wheel rotation centre. Fork lifting
# moves the mast/frame roughly 3 cm toward the rotation centre.
UP_LIDAR_X = UP_LIDAR_X_FORKS_DOWN_M
UP_LIDAR_X_FORKS_UP = UP_LIDAR_X_FORKS_UP_M
SIM_UP_LIDAR_X = UP_LIDAR_X # Gazebo must reproduce the real robot
UP_LIDAR_Z = 1.645          # canonical upper-LiDAR height
# Compatibility coordinates for real logs recorded before the odometry origin
# was moved to the rear-wheel rotation centre. This is not a physical mount.
LEGACY_REAL_UP_LIDAR_X = UP_LIDAR_X - DRIVE_WHEEL_X_M
LEGACY_REAL_UP_LIDAR_X_FORKS_UP = UP_LIDAR_X_FORKS_UP - DRIVE_WHEEL_X_M
LEGACY_REAL_FRAME_LAST_DATE = '20260814'

# Full SE2/Z poses (x, y, z, yaw) in rotation-center frame.
CAMERA_POSE_BODY = {
    'left': (LOW_LIDAR_X, LOW_LIDAR_HALF_Y, LOW_LIDAR_Z, LOW_LIDAR_YAW),
    'right': (LOW_LIDAR_X, -LOW_LIDAR_HALF_Y, LOW_LIDAR_Z, -LOW_LIDAR_YAW),
    'up': (UP_LIDAR_X, 0.0, UP_LIDAR_Z, 0.0),
}
# Wrong poses used before 2026-08 fix (fork-tip offset, yaw=0) — for reprocessing.
LEGACY_CAMERA_POSE_BODY = {
    'left': (0.33, 0.378, 0.10, 0.0),
    'right': (0.33, -0.378, 0.10, 0.0),
    'up': (-0.26, 0.0, 1.64, 0.0),
}
# XY mount for near-sensor filtering (drop body self-hits).
CAMERA_EXTRINSICS = {
    name: (pose[0], pose[1]) for name, pose in CAMERA_POSE_BODY.items()
}
# MAP_CAMERAS = ['left', 'right', 'up']  # e.g. ['left', 'right'] or ['up']
MAP_CAMERAS = ['up']

# Input and output filtering
MIN_RANGE = 0.4
SENSOR_MIN_RANGE = 0.15  # drop (0,0) / body-blocked returns near lidar
MAX_RANGE = 12.0
MATCH_MIN_RANGE = 1.0
MAP_CELL = 0.05
MAP_MIN_HITS = 5  # default improved map: suppress single-pass ghosts
MAP_MIN_COMPONENT_CELLS = 50
# Drop frames from hit-grid when |enc_wz| (or |Δyaw|/dt) exceeds this.
MAP_SKIP_TURN_WZ = 0.15
# Synthesize per-beam times from point bearing when recorder stored identical stamps.
SYNTHETIC_SCAN_PERIOD_S = 0.1  # 10 Hz lidar

# Robust ICP
ICP_ITERS = 18
ICP_POINTS = 280
ICP_MAX_DIST = 0.55
ICP_TRIM = 0.70
ICP_MIN_INLIERS = 45
STATIC_XY = 0.018
STATIC_TH = math.radians(0.35)
MAX_FRAME_XY = 0.30
MAX_FRAME_TH = math.radians(4.0)

# Local submap and keyframes
SUBMAP_KEYFRAMES = 10
KEYFRAME_XY = 0.18
KEYFRAME_TH = math.radians(5.0)
# Frontend / sequential ICP quality gates (factor LiDAR hardening).
FRONTEND_SCAN_MIN_OVERLAP = 0.35
FRONTEND_SCAN_MAX_RMSE = 0.08          # m; reject scan delta if worse
FRONTEND_SUBMAP_MAX_DIST = 0.45        # m; scan-to-submap ICP
FRONTEND_WHEEL_AGREE_XY = 0.25         # m; scan vs wheel disagreement
FRONTEND_WHEEL_AGREE_TH = math.radians(12.0)
SEQ_MIN_OVERLAP = 0.38
SEQ_MAX_RMSE = 0.08                    # m; sequential graph edge gate
SEQ_REQUIRE_WHEEL_AGREE = True         # drop seq edge if disagrees with wheel
# Sequential keyframes are normally ~0.18 m apart.  The generic 0.25 m
# frontend gate would therefore accept the degenerate aisle solution "no
# motion". Keep LiDAR corrections local around the encoder prediction.
SEQ_WHEEL_AGREE_XY = 0.08
SEQ_WHEEL_AGREE_TH = math.radians(5.0)
SEQ_WHEEL_DISTANCE_RATIO_MIN = 0.65
SEQ_WHEEL_DISTANCE_RATIO_MAX = 1.35
LOOP_RADIUS = 1.3
LOOP_FRAME_GAP = 120
LOOP_MAX_ERROR = 0.14
LOOP_MIN_OVERLAP = 0.50
LOOP_MAX_CORR_XY = 0.55
LOOP_MAX_CORR_TH = math.radians(15.0)
LOOP_BEST_OF_K = 3                     # try top-K spatial candidates, keep best
# Aggressive preset (experiments / poor coverage runs)
LOOP_RADIUS_AGGRO = 8.0
LOOP_FRAME_GAP_AGGRO = 40
LOOP_MAX_ERROR_AGGRO = 0.40
LOOP_MIN_OVERLAP_AGGRO = 0.12
LOOP_MAX_CORR_XY_AGGRO = 2.50
LOOP_MAX_CORR_TH_AGGRO = math.radians(35.0)
POSE_GRAPH_SPARSE_JACOBIAN = True

# LiDAR points with robot timestamps (/raw_points: x, y, t)
POINTS_PATH = '/raw_points'
POINTS_PATH_FALLBACK = '/points'

# Wheel encoder odometry (/check_logs: enc_t, enc_v, enc_dgr)
ENCODER_PATH = '/check_logs'
ENCODER_PATH_FALLBACK = '/chek_logs'
ENCODER_UTC_OFFSET_HOURS = 3
WHEEL_SPEED_SCALE = -1.0
WHEEL_DIRECTION_SIGN = 1.0
WHEELBASE_M = DRIVE_WHEEL_X_M
# Production src convention verified against nav_* GT: body_vx=-enc_v;
# steering 0° is straight, -90° right, +90° left.
WHEEL_DIRECTION_OFFSET = 0.0
WHEEL_TRANS_WEIGHT = 0.20
WHEEL_ROT_WEIGHT = 0.30
WHEEL_EDGE_WEIGHT = 0.10
WHEEL_MAX_SYNC_ERROR = 0.35
WHEEL_MAX_TRANS_DISAGREEMENT = 0.12
WHEEL_MAX_ROT_DISAGREEMENT = math.radians(3.0)
# Factor-graph (LIO-SAM-lite): soft odom σ and lidar/odom delay search.
FACTOR_ODOM_SIGMA_XY = 0.08          # m; lower → trust wheel more
FACTOR_ODOM_SIGMA_YAW = math.radians(2.5)
FACTOR_LIDAR_BASE_WEIGHT = 1.0
FACTOR_LOOP_BASE_WEIGHT = 4.0
# Diagnostics: joint s_v and auto δt did not help on warehouse GT → off.
FACTOR_ESTIMATE_SCALE = False        # keep s_v = 1.0
FACTOR_SCALE_PRIOR_SIGMA = 0.05      # soft prior if scale re-enabled
FACTOR_CALIBRATE_DT = False          # use raw lidar/encoder timestamps
FACTOR_DT_MIN = -0.08                # s; encoder lag relative to lidar
FACTOR_DT_MAX = 0.08
FACTOR_DT_STEP = 0.01
FACTOR_DT_MAX_PAIRS = 100
# Sequential ICP target: 'scan' | 'submap' (last SUBMAP_KEYFRAMES).
# Full-run: scan-only frontend collapses (~8 m ATE); submap N=10 best FPS/ATE.
FACTOR_LIDAR_SEQ_TARGET = 'submap'
ENCODER_STATIC_V = 0.008
POINT_DESKEW_MAX_DT = 0.15
POINT_DESKEW_MAX_CORRECTION = 0.15
POINT_DESKEW_MAX_ANGLE = math.radians(3.0)
# Default deskew: wheel translation then LiDAR yaw; 'full_se2' uses pose SE2.
# Motion-accuracy winner on run_20260807_164805 (up): full SE2 deskew.
DESKEW_MODE = 'full_se2'  # none | wheel | wheel+yaw | wheel+yaw_rate | full_se2
TURN_KEYFRAME_XY = 0.10
TURN_KEYFRAME_TH = math.radians(2.5)

# Legacy compatibility flag. Mapping code deliberately ignores it: /odom is
# evaluation ground truth only; poses always come from enc_v, enc_dgr and LiDAR.
USE_ABSOLUTE_ODOM = False
# Pallet-jack lift. Forks rise 10–20 cm (sim 15). Rear wheels sit on a bogie
# (основа), not on the tines; as forks go up the bogie rolls toward the body
# and origin→body shrinks to 93 cm. Body-frame lidar X then shrinks by
# FORK_LIFT_BODY_DX (use when mapping with a raised pallet).
FORK_LIFT_H = 0.15
ORIGIN_TO_BODY_DOWN = 0.975  # m, forks down (insert)
ORIGIN_TO_BODY_UP = 0.945     # m, forks up (carry, 15 cm lift)
FORK_LIFT_BODY_DX = ORIGIN_TO_BODY_DOWN - ORIGIN_TO_BODY_UP

# Mapping / odometry origin is already the rear-fork ICR.  A pure spin must
# therefore have zero origin translation.  The historical -1.235 value mixed
# the old body-centred SDF frame with the current rear-axle frame and injected
# a false lateral displacement into simulated wheel odometry.
REAR_PIVOT_X = 0.0

session = requests.Session()
session.verify = False
session.trust_env = False

METHOD_PRESETS = {
    'clouds': dict(
        use_encoder=False,
        deskew_mode='none',
        synthesize_beam_times=False,
        skip_turn_wz=None,
        map_min_hits=3,
    ),
    'odom': dict(
        use_encoder=True,
        deskew_mode='none',
        synthesize_beam_times=False,
        skip_turn_wz=None,
        # ``odom`` is the raw, reproducible wheel baseline.  It must not be
        # altered by appearance-based loop closures; use factor_no_loops or a
        # separately validated SLAM profile when scan constraints are desired.
        skip_loop_closures=True,
        map_min_hits=3,
    ),
    'improved': dict(
        use_encoder=True,
        deskew_mode='full_se2',
        synthesize_beam_times=True,
        skip_turn_wz=0.15,
        map_min_hits=5,
    ),
    # Soft wheel + scan-to-submap ICP + LiDAR loops; s_v=1; no δt calib.
    'factor': dict(
        use_encoder=True,
        deskew_mode='full_se2',
        synthesize_beam_times=True,
        factor_mode=True,
        skip_turn_wz=0.15,
        map_min_hits=5,
        estimate_scale=False,
        calibrate_dt=False,
        frontend_icp='submap',
        lidar_seq_target='submap',
        submap_keyframes=10,
    ),
    # Same estimator as factor, but with global loop closures explicitly off.
    # This is the safe warehouse baseline while repeated-aisle loop validation
    # is being hardened. Sequential LiDAR-to-submap constraints remain active.
    'factor_no_loops': dict(
        use_encoder=True,
        deskew_mode='full_se2',
        synthesize_beam_times=True,
        factor_mode=True,
        skip_loop_closures=True,
        skip_turn_wz=0.15,
        map_min_hits=5,
        estimate_scale=False,
        calibrate_dt=False,
        frontend_icp='submap',
        lidar_seq_target='submap',
        submap_keyframes=10,
    ),
    # Adaptive wheel prior + scan-to-submap ICP with anisotropic information
    # and loop closures.  Wheel data is only a prediction/soft graph factor.
    'odom_fixed_v2': dict(
        use_encoder=True,
        deskew_mode='full_se2',
        synthesize_beam_times=True,
        factor_mode=True,
        skip_turn_wz=0.15,
        map_min_hits=5,
        estimate_scale=True,
        calibrate_dt=False,
        frontend_icp='submap',
        lidar_seq_target='submap',
        submap_keyframes=20,
    ),
    'submap': dict(
        use_encoder=True,
        deskew_mode='full_se2',
        synthesize_beam_times=True,
        submap_mode=True,
        skip_turn_wz=0.15,
        map_min_hits=5,
    ),
}

SOURCE_PRESETS = {
    'sim': dict(use_sim_time=True),
    'real': dict(use_sim_time=False),
    'auto': dict(use_sim_time=None),
}
