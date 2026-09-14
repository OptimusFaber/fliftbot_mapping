"""Offline/global mapping orchestration facade."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from .config import (
    DESKEW_MODE, HZ, KEYFRAME_TH, KEYFRAME_XY, MAP_CAMERAS,
    MAP_CELL, MAP_MIN_HITS, MAP_SKIP_TURN_WZ, REAR_PIVOT_X, ROOT,
    SIM_UP_LIDAR_X, SYNTHETIC_SCAN_PERIOD_S, TURN_KEYFRAME_TH, TURN_KEYFRAME_XY,
    LEGACY_REAL_FRAME_LAST_DATE, LEGACY_REAL_UP_LIDAR_X, WHEELBASE_M,
    LEGACY_REAL_UP_LIDAR_X_FORKS_UP, METHOD_PRESETS, UP_LIDAR_X,
    UP_LIDAR_X_FORKS_UP,
)
from .deskew import (
    deskew_all_frames, ensure_beam_point_times, full_se2_deskew_from_poses,
    rotational_deskew_from_poses,
)
from .encoder_sync import (
    build_keyframes_from_poses, build_trajectory_from_odom,
    calibrate_lidar_odom_delay, get_wheel_se2, load_frame_timestamp,
    samples_have_usable_steering,
    set_wheel_direction_offset, set_wheel_se2,
    shift_encoder_samples, synchronize_encoders,
)
from .geometry_icp import motion_from_scans, normalize_angle, snap_map_to_manhattan
from .io_frames import (
    combine_camera_clouds, format_point_timestamp, list_run_frame_paths,
    load_horizon_npz_frame, normalize_camera_list, points_from_frame_record,
    shift_up_body_x,
)
from .map_filtering import build_global_map, build_global_map_filtered
from .output import save_global_map, save_horizon_npz, save_local_frame
from .source_io import (
    append_encoder_sample,
    collect_run_encoder_samples, encoder_log_path, fetch_encoder_log,
    fetch_raw_points, load_encoder_record, resolve_base_url, set_base_url,
)
from . import source_io as _source_io
from flifltbot.common.runtime.npz_log import AsyncNpzWriter
from .trajectory_graph import (
    apply_pose_graph_corrections, build_pose_graph_edges,
    estimate_initial_trajectory, optimize_keyframe_poses, refine_poses_to_map,
)
from .submap_slam import build_submap_trajectory
from flifltbot.common.output.layout import make_run_directories

FORBIDDEN_MAPPING_MOTION_FIELDS = frozenset(
    ('odom_x', 'odom_y', 'odom_yaw', 'enc_wz')
)


def normalize_http_lidar_timestamps(camera_clouds, *, host_time_s=None,
                                     offset_s=None):
    """Put robot HTTP LiDAR timestamps on the encoder Unix-time axis.

    Some robot images expose ``/raw_points`` ``t`` as seconds since boot
    (typically a few thousand), while ``/check_logs`` uses wall-clock Unix
    timestamps.  Leaving those domains mixed makes every encoder/LiDAR pair
    unsynchronizable and produces 1970-dated files.  Detect the boot-time
    domain conservatively (<1e8) and anchor it once to the local HTTP receive
    time.  Unix timestamps are left byte-for-byte unchanged.

    Returns ``(offset_s, domain)`` where domain is ``"unix"`` or
    ``"monotonic_host_anchored"``.
    """
    host_time_s = time.time() if host_time_s is None else float(host_time_s)
    finite_stamps = []
    for cloud in camera_clouds.values():
        try:
            stamp = float(cloud.get('t'))
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(stamp):
            finite_stamps.append(stamp)
    if not finite_stamps:
        return offset_s, 'unix'
    monotonic_domain = max(abs(value) for value in finite_stamps) < 1.0e8
    if not monotonic_domain:
        return offset_s, 'unix'
    if offset_s is None:
        offset_s = host_time_s - float(np.median(finite_stamps))
    for cloud in camera_clouds.values():
        try:
            stamp = float(cloud.get('t'))
        except (AttributeError, TypeError, ValueError):
            stamp = float('nan')
        if np.isfinite(stamp):
            cloud['t'] = stamp + offset_s
        point_times = cloud.get('point_times')
        if point_times is not None:
            values = np.asarray(point_times, dtype=np.float64)
            values = values.copy()
            finite = np.isfinite(values)
            values[finite] += offset_s
            cloud['point_times'] = values
    return offset_s, 'monotonic_host_anchored'


def sanitize_mapping_encoder_sample(sample):
    """Remove signals that are forbidden as mapping motion inputs."""
    if sample is None:
        return None
    return {
        key: value for key, value in sample.items()
        if key not in FORBIDDEN_MAPPING_MOTION_FIELDS
    }


def scaled_min_hits_for_cell(cell_m, base_hits=None, base_cell=None):
    """Lower hit threshold for finer grids so walls are not thinned away.

    Walls are ~1D: hit density per cell scales roughly with cell size, not area.
    """
    base_hits = MAP_MIN_HITS if base_hits is None else int(base_hits)
    base_cell = MAP_CELL if base_cell is None else float(base_cell)
    cell_m = float(cell_m)
    return max(1, int(round(base_hits * (cell_m / base_cell))))


def build_global_map_from_run(
    run_dir, method='improved', cameras=None, tag=None, source='auto',
    cell_m=None, map_min_hits=None, min_component_cells=None, write_module_copy=False,
    **extra,
):
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f'run dir not found: {run}')
    if method not in METHOD_PRESETS:
        raise ValueError(f'unknown method {method!r}')
    kwargs = dict(METHOD_PRESETS[method])
    kwargs.update(extra)
    fork_state = str(kwargs.pop('fork_state', 'auto')).lower()
    if fork_state == 'auto':
        fork_state = _recording_fork_state(run)
    if fork_state not in ('down', 'up'):
        raise ValueError("fork_state must be 'auto', 'down' or 'up'")
    geometry_profile = 'explicit'
    if 'up_body_x' not in extra:
        inferred_x, geometry_profile = _recording_up_body_x(
            run, source, fork_state=fork_state,
        )
        if inferred_x is not None:
            kwargs['up_body_x'] = inferred_x
    if cameras:
        kwargs['cameras'] = cameras
    if cell_m is not None:
        kwargs['cell_m'] = cell_m
        if map_min_hits is None:
            base = kwargs.get('map_min_hits', MAP_MIN_HITS)
            kwargs['map_min_hits'] = scaled_min_hits_for_cell(cell_m, base_hits=base)
        else:
            kwargs['map_min_hits'] = int(map_min_hits)
    elif map_min_hits is not None:
        kwargs['map_min_hits'] = int(map_min_hits)
    if min_component_cells is not None:
        kwargs['min_component_cells'] = min_component_cells
    points, poses, files, stats = sew_all_local_frames(
        run, source=source, **kwargs
    )
    stats['recording_geometry_profile'] = geometry_profile
    stats['fork_state'] = fork_state
    if not files:
        raise FileNotFoundError(
            f'no frame JSON or horizon/*.npz files found in {run} '
            f'(expected names like 00001_....json or horizon/00001_....npz)'
        )
    out_tag = tag or (run.name if run.name.startswith('run_') else f'run_{run.name}')
    if method != 'improved' and not str(out_tag).endswith(f'_{method}'):
        out_tag = f'{out_tag}_{method}'
    used_encoder = bool(stats.get('encoder_frames')) and kwargs.get('use_encoder', True)
    out_json, out_png = save_global_map(
        points, poses, out_tag, run, used_encoder=used_encoder, source=source,
        cell_m=cell_m, write_module_copy=write_module_copy,
    )
    return {
        'points': points, 'poses': poses, 'files': files, 'stats': stats,
        'out_json': out_json, 'out_png': out_png, 'method': method, 'run_dir': run,
    }

def build_snapshot_map(frame_path, cameras=None, tag='snapshot'):
    frame_file = Path(frame_path)
    payload = json.loads(frame_file.read_text(encoding='utf-8'))
    points, frame_t, point_times, used_cameras = points_from_frame_record(payload, cameras=cameras)
    safe_ts = format_point_timestamp(frame_t)
    local_dir = frame_file.parent
    save_local_frame(local_dir, 1, safe_ts, points, frame_t=frame_t, point_times=point_times, map_cameras=used_cameras)
    stem = f"{1:05d}_{safe_ts.replace(':', '-').replace('+00:00', 'Z')}"
    return {
        'points': points, 'frame_t': frame_t, 'point_times': point_times, 'used_cameras': used_cameras,
        'out_json': local_dir / f'{stem}.json', 'out_png': local_dir / f'{stem}.png',
    }

def collect_and_build(run_dir, method='improved', cameras=None):
    return build_global_map_from_run(run_dir, method=method, cameras=cameras)

def _infer_run_source(local_dir, source='auto'):
    if source in ('sim', 'real'):
        return source
    parts = Path(local_dir).parts
    if 'sim' in parts:
        return 'sim'
    if 'real' in parts:
        return 'real'
    return source


def _recording_date(local_dir):
    """Return YYYYMMDD from the run path, or from its first frame name."""
    path = Path(local_dir)
    for part in reversed(path.parts):
        match = re.search(r'(?<!\d)(20\d{6})(?!\d)', part)
        if match:
            return match.group(1)
    for frame_path in list_run_frame_paths(path)[:1]:
        match = re.search(r'(20\d{2})-(\d{2})-(\d{2})', frame_path.name)
        if match:
            return ''.join(match.groups())
    return None


def _recording_fork_state(local_dir):
    """Read recorded fork state; old runs without metadata default to down."""
    meta_path = Path(local_dir) / 'run_meta.json'
    if meta_path.is_file():
        try:
            state = str(json.loads(meta_path.read_text()).get('fork_state', '')).lower()
            if state in ('down', 'up'):
                return state
        except (OSError, ValueError, TypeError):
            pass
    return 'down'


def _recording_up_body_x(local_dir, source='auto', fork_state='down'):
    """Select the up-LiDAR lever arm in the coordinate frame stored by a log."""
    resolved_source = _infer_run_source(local_dir, source)
    raised = str(fork_state).lower() == 'up'
    if resolved_source == 'sim':
        return (
            UP_LIDAR_X_FORKS_UP if raised else SIM_UP_LIDAR_X,
            'sim_rotation_centre_forks_up' if raised else 'sim_rotation_centre',
        )
    if resolved_source != 'real':
        return None, 'auto_unknown'
    recording_date = _recording_date(local_dir)
    if recording_date and recording_date <= LEGACY_REAL_FRAME_LAST_DATE:
        return (
            LEGACY_REAL_UP_LIDAR_X_FORKS_UP
            if raised else LEGACY_REAL_UP_LIDAR_X,
            'legacy_real_body_origin_forks_up'
            if raised else 'legacy_real_body_origin',
        )
    return (
        UP_LIDAR_X_FORKS_UP if raised else UP_LIDAR_X,
        'real_rotation_centre_forks_up' if raised else 'real_rotation_centre',
    )


def sew_all_local_frames(
    local_dir,
    use_encoder=True,
    max_frames=None,
    deduplicate_timestamps=False,
    skip_duplicate_pair_icp=True,
    deskew_points=True,
    cameras=None,
    deskew_mode=None,
    keyframe_xy=None,
    keyframe_th=None,
    map_builder='hit_grid',
    adaptive_turn_keyframes=False,
    icp_max_xy=0.35,
    icp_max_th=None,
    icp_min_overlap=0.42,
    return_internals=False,
    synthesize_beam_times=True,
    skip_turn_wz=MAP_SKIP_TURN_WZ,
    map_min_hits=None,
    cell_m=None,
    min_component_cells=None,
    source='auto',
    up_body_x=None,
    factor_mode=False,
    estimate_scale=None,
    skip_lidar_sequential=False,
    skip_loop_closures=False,
    calibrate_dt=None,
    encoder_dt_s=None,
    frontend_icp='auto',
    lidar_seq_target=None,
    submap_keyframes=None,
    submap_mode=False,
):
    from .config import (
        FACTOR_CALIBRATE_DT, FACTOR_ESTIMATE_SCALE, FACTOR_LIDAR_SEQ_TARGET,
    )

    total_started = time.perf_counter()
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    factor_mode = bool(factor_mode)
    requested_factor_mode = factor_mode
    submap_mode = bool(submap_mode)
    if estimate_scale is None:
        estimate_scale = FACTOR_ESTIMATE_SCALE if factor_mode else False
    else:
        estimate_scale = bool(estimate_scale)
    if calibrate_dt is None:
        calibrate_dt = bool(FACTOR_CALIBRATE_DT) if factor_mode else False
    else:
        calibrate_dt = bool(calibrate_dt)
    skip_lidar_sequential = bool(skip_lidar_sequential)
    skip_loop_closures = bool(skip_loop_closures)
    frontend_icp = str(frontend_icp or 'auto').lower()
    if lidar_seq_target is None:
        lidar_seq_target = FACTOR_LIDAR_SEQ_TARGET if factor_mode else 'scan'
    lidar_seq_target = str(lidar_seq_target).lower()
    cameras = normalize_camera_list(cameras or MAP_CAMERAS)
    resolved_source = _infer_run_source(local_dir, source)
    # The anisotropic scan-to-submap factor frontend is beneficial for dense,
    # regular Gazebo scans, but it over-constrains sparse real scans and makes
    # parallel wall observations split into thick bands. Real factor mapping
    # uses the robust scan factors (scalar weight + wheel-agreement gate) that
    # are stable on the measured room runs. The optimizer remains a pose graph.
    graph_factor_mode = factor_mode and resolved_source != 'real'
    factor_constraint_profile = (
        'sim_anisotropic_submap'
        if graph_factor_mode
        else ('real_robust_scan' if factor_mode else 'standard')
    )
    if up_body_x is None and resolved_source == 'sim':
        up_body_x = SIM_UP_LIDAR_X
    if (
        up_body_x is not None
        and 'up' in cameras
        and abs(float(up_body_x) - UP_LIDAR_X) > 1e-9
    ):
        print(
            f'reprojecting recorded up body x {UP_LIDAR_X:.2f} -> '
            f'{float(up_body_x):.2f} m (recording geometry profile)'
        )
    if deskew_mode is None:
        if not deskew_points:
            deskew_mode = 'none'
        else:
            deskew_mode = DESKEW_MODE
    deskew_mode = str(deskew_mode).lower()
    keyframe_xy = KEYFRAME_XY if keyframe_xy is None else keyframe_xy
    keyframe_th = KEYFRAME_TH if keyframe_th is None else keyframe_th
    icp_max_th = math.radians(8.0) if icp_max_th is None else icp_max_th
    files = list_run_frame_paths(local_dir)
    if not files:
        return [], [], [], {}

    if max_frames is not None:
        files = files[:max_frames]

    frames = []
    metadata = []
    legacy_frames = 0
    skipped_sparse = 0
    for path in files:
        if path.suffix.lower() == '.npz':
            loaded = load_horizon_npz_frame(path)
            if loaded is None:
                skipped_sparse += 1
                continue
            points, frame_t, point_times, used_cameras, data = loaded
            # Pandar horizon NPZ stores XY in the sensor frame.  Treat it as
            # the upper LiDAR when applying the body-frame lever arm; the old
            # check only matched ``up`` and silently placed Pandar at (0, 0).
            if up_body_x is not None and any(
                name in ('up', 'top', 'horizon') for name in used_cameras
            ):
                points = shift_up_body_x(
                    points, recorded_x=0.0, target_x=up_body_x
                )
                data['map_cameras'] = ['up']
        else:
            data = json.loads(path.read_text(encoding='utf-8'))
            points, frame_t, point_times, used_cameras = points_from_frame_record(
                data, cameras=cameras, up_body_x=up_body_x
            )
            if frame_t is not None:
                data['t'] = frame_t
            data['point_times'] = [
                None if not np.isfinite(value) else float(value)
                for value in point_times
            ]
            data['map_cameras'] = used_cameras
        if used_cameras == ['legacy']:
            legacy_frames += 1
        frames.append(points)
        metadata.append(data)
    if skipped_sparse:
        print(f'skipped {skipped_sparse} sparse horizon frames (<200 pts)')
    if not frames:
        return [], [], [], {}

    if synthesize_beam_times:
        n_synth = ensure_beam_point_times(frames, metadata)
        if n_synth:
            print(
                f'synthesized per-beam timestamps for {n_synth}/{len(frames)} '
                'frames (period inferred from consecutive cloud stamps)'
            )

    encoder_samples = collect_run_encoder_samples(metadata, local_dir)
    # Mapping input contract: wheel speed + measured steering angle only.
    # Old simulation recordings may additionally contain Gazebo ground-truth
    # pose and body yaw-rate.  Strip all of it before synchronization, deskew,
    # trajectory estimation and factor construction.  LiDAR plus enc_v and
    # enc_dgr are the only allowed motion inputs.
    encoder_samples = [
        sanitize_mapping_encoder_sample(sample) for sample in encoder_samples
    ]
    encoder_frames = len(encoder_samples)
    # Reset the physical steering convention. Both src and the calibrated
    # recorder use 0° straight, +left, -right. Absolute odom is not used.
    set_wheel_direction_offset(None)
    set_wheel_se2(False)
    if use_encoder and encoder_samples:
        # Sim steering is already expressed in the robot frame.  Real logs use
        # the configured hardware convention.  Neither choice is calibrated
        # against a global pose.
        calibrated = 0.0 if resolved_source == 'sim' else None
        if calibrated is not None:
            set_wheel_direction_offset(calibrated)
            if resolved_source == 'sim':
                print(
                    f'sim wheel direction offset={math.degrees(calibrated):.1f}° '
                    '(known robot-frame convention; no global /odom used)'
                )
            else:
                print(
                    f'calibrated wheel direction offset={math.degrees(calibrated):.1f}° '
                    '(enc_dgr→body)'
                )
        use_wheel_se2 = samples_have_usable_steering(encoder_samples)
        set_wheel_se2(use_wheel_se2)
        if use_wheel_se2:
            print(
                'wheel speed + actual steering usable: tricycle SE(2) '
                f'(wheelbase={WHEELBASE_M:.3f} m) + LiDAR; no absolute /odom'
            )
    dt_stats = {
        'best_dt_s': 0.0,
        'best_cost': None,
        'half_width_s': None,
        'n_pairs': 0,
        'curve': [],
    }
    if use_encoder and encoder_dt_s is not None and encoder_samples:
        fixed_dt = float(encoder_dt_s)
        encoder_samples = shift_encoder_samples(encoder_samples, fixed_dt)
        dt_stats = {
            'best_dt_s': fixed_dt,
            'best_cost': None,
            'half_width_s': None,
            'n_pairs': 0,
            'curve': [],
            'fixed': True,
        }
        print(f'using fixed lidar↔encoder δt={fixed_dt*1000:.1f} ms')
    elif use_encoder and calibrate_dt and encoder_samples:
        dt_stats = calibrate_lidar_odom_delay(
            metadata, encoder_samples, frames=frames,
        )
        best_dt = float(dt_stats.get('best_dt_s') or 0.0)
        if abs(best_dt) > 1e-6:
            encoder_samples = shift_encoder_samples(encoder_samples, best_dt)
    encoders, sync_stats = synchronize_encoders(
        metadata, encoder_samples=encoder_samples
    )
    # Long, repeated warehouse traversals produce many visually convincing but
    # false loop closures between near-identical aisles.  Keep sequential
    # LiDAR-to-submap constraints: they are what correct local wheel drift.
    # Only speculative global loop closures are disabled by this guard.
    long_run_sensor_guard = bool(
        use_encoder
        and get_wheel_se2()
        and resolved_source == 'sim'
        and len(frames) >= 8000
    )
    if long_run_sensor_guard:
        skip_loop_closures = True
        if requested_factor_mode:
            factor_constraint_profile = 'long_run_wheel_lidar_submap_no_loops'
            # Wheel telemetry owns the accumulated motion. LiDAR constraints
            # are still built below, but are deliberately weak corrections;
            # per-frame scan yaw has a small systematic aisle bias that must
            # not replace the encoder trajectory in the frontend.
            frontend_icp = 'wheel'
        print(
            'long-run warehouse guard: disabling ambiguous aisle loop closures; '
            'keeping wheel priors and sequential LiDAR-to-submap constraints'
        )
    deskew_stats = {
        'deskewed_frames': 0,
        'deskewed_points': 0,
        'deskew_max_correction_m': 0.0,
        'deskew_mode': deskew_mode,
    }
    do_wheel_deskew = deskew_mode in (
        'wheel', 'wheel+yaw', 'wheel+yaw_rate', 'full_se2'
    )
    if use_encoder and do_wheel_deskew and deskew_mode != 'full_se2':
        frames, wheel_stats = deskew_all_frames(
            frames, metadata, encoder_samples=encoder_samples
        )
        deskew_stats.update(wheel_stats)
        deskew_stats['deskew_mode'] = deskew_mode
    loaded_at = time.perf_counter()

    pose_indices = []
    original_to_pose = []
    previous_t = None
    for index, data in enumerate(metadata):
        frame_t = load_frame_timestamp(data)
        is_duplicate = (
            bool(pose_indices)
            and deduplicate_timestamps
            and frame_t is not None
            and frame_t == previous_t
        )
        if not is_duplicate:
            pose_indices.append(index)
        original_to_pose.append(len(pose_indices) - 1)
        previous_t = frame_t

    pose_frames = [frames[index] for index in pose_indices]
    # Defense in depth: synchronized pose records must contain no forbidden
    # Gazebo-derived motion fields either.
    pose_encoders = [
        sanitize_mapping_encoder_sample(encoders[index])
        for index in pose_indices
    ]
    duplicate_frames = len(frames) - len(pose_frames)
    if duplicate_frames:
        print(
            f'pose estimation: {len(pose_frames)} unique timestamped scans; '
            f'skipping {duplicate_frames} exact repeats'
        )

    print(f'map cameras: {", ".join(cameras)}; deskew={deskew_mode}')
    if legacy_frames:
        print(
            f'note: {legacy_frames}/{len(frames)} frames are legacy '
            '(combined local_points only; per-camera select unavailable)'
        )
    if encoder_frames and use_encoder:
        print(
            'estimating poses with timestamp-synchronized wheel odometry: '
            f'{sync_stats["synced_frames"]}/{len(frames)} frames, '
            f'{sync_stats["unique_samples"]} unique encoder samples, '
            f'median |dt|={sync_stats["median_sync_error_s"]!s}s; '
            f'deskewed {deskew_stats.get("deskewed_points", 0)} points in '
            f'{deskew_stats.get("deskewed_frames", 0)} frames'
        )
    else:
        print('estimating poses with LiDAR only...')
    traj_kwargs = dict(
        use_encoder=use_encoder,
        source_indices=pose_indices,
        skip_duplicate_pair_icp=skip_duplicate_pair_icp,
        keyframe_xy=keyframe_xy,
        keyframe_th=keyframe_th,
        adaptive_turn_keyframes=adaptive_turn_keyframes,
        icp_max_xy=icp_max_xy,
        icp_max_th=icp_max_th,
        icp_min_overlap=icp_min_overlap,
        cell_m=cell_m,
        factor_mode=graph_factor_mode,
        frontend_icp=frontend_icp,
        submap_keyframes=submap_keyframes,
    )
    graph_kwargs = dict(
        use_encoder=use_encoder,
        loop_aggro=False,
        factor_mode=graph_factor_mode,
        estimate_scale=estimate_scale,
        skip_lidar_sequential=skip_lidar_sequential,
        skip_loop_closures=skip_loop_closures,
        lidar_seq_target=lidar_seq_target,
        submap_keyframes=submap_keyframes,
        preserve_wheel_translation=bool(
            long_run_sensor_guard and graph_factor_mode
        ),
    )
    scale_stats = {'s_v': 1.0, 'estimate_scale': False}
    submap_stats = {}
    if submap_mode:
        if not (use_encoder and get_wheel_se2()):
            raise ValueError(
                'submap method requires usable enc_v + enc_dgr'
            )
        wheel_poses, _ = build_trajectory_from_odom(pose_encoders)
        unique_poses, submap_stats = build_submap_trajectory(
            pose_frames, wheel_poses,
        )
        keyframes = build_keyframes_from_poses(
            pose_frames, pose_encoders, unique_poses, pose_indices,
            keyframe_xy=keyframe_xy, keyframe_th=keyframe_th,
        )
        poses = [unique_poses[index] for index in original_to_pose]
        optimized = [list(item['pose']) for item in keyframes]
        initial_poses = unique_poses
        graph_cost = 0.0
        graph_stats = {
            'wheel_edges': max(0, len(keyframes) - 1),
            'lidar_seq_edges': submap_stats.get('submap_icp_accepted', 0),
            'loop_closures': 0,
            'keyframes': len(keyframes),
        }
        factor_constraint_profile = 'overlapping_distance_submaps'
        print(
            f'submap trajectory: {submap_stats["submaps"]} windows, '
            f'ICP accepted={submap_stats["submap_icp_accepted"]}, '
            f'rejected={submap_stats["submap_icp_rejected"]}'
        )
    else:
        if use_encoder:
            print(
                'forbidden GT/body-rate fields removed; '
                'using bicycle-model enc_v + enc_dgr + LiDAR only'
            )
        initial_poses, keyframes = estimate_initial_trajectory(
            pose_frames,
            pose_encoders,
            **traj_kwargs,
        )
        edges, graph_stats = build_pose_graph_edges(keyframes, **graph_kwargs)
        optimized, graph_cost, scale_stats = optimize_keyframe_poses(
            keyframes, edges,
            estimate_scale=estimate_scale and graph_factor_mode,
        )
        unique_poses = apply_pose_graph_corrections(
            initial_poses, keyframes, optimized
        )
        poses = [unique_poses[index] for index in original_to_pose]
    trajectory_at = time.perf_counter()
    rotation_deskew_stats = {
        'rotational_deskewed_frames': 0,
        'rotational_deskewed_points': 0,
        'deskew_max_angle_deg': 0.0,
        'se2_deskewed_frames': 0,
        'se2_deskewed_points': 0,
    }

    def _reestimate_after_deskew():
        nonlocal frames, poses, keyframes, graph_stats, graph_cost
        nonlocal initial_poses, optimized, unique_poses
        pose_frames_local = [frames[index] for index in pose_indices]
        nonlocal scale_stats
        if submap_mode:
            wheel_poses_local, _ = build_trajectory_from_odom(pose_encoders)
            unique_poses, updated = build_submap_trajectory(
                pose_frames_local, wheel_poses_local,
            )
            submap_stats.update(updated)
            keyframes = build_keyframes_from_poses(
                pose_frames_local, pose_encoders, unique_poses, pose_indices,
                keyframe_xy=keyframe_xy, keyframe_th=keyframe_th,
            )
            initial_poses = unique_poses
            optimized = [list(item['pose']) for item in keyframes]
            graph_cost = 0.0
            graph_stats = {
                'wheel_edges': max(0, len(keyframes) - 1),
                'lidar_seq_edges': updated.get('submap_icp_accepted', 0),
                'loop_closures': 0,
                'keyframes': len(keyframes),
            }
            poses = [unique_poses[index] for index in original_to_pose]
            return poses
        initial_poses, keyframes = estimate_initial_trajectory(
            pose_frames_local,
            pose_encoders,
            **traj_kwargs,
        )
        edges_local, graph_stats = build_pose_graph_edges(
            keyframes, **graph_kwargs
        )
        optimized, graph_cost, scale_stats = optimize_keyframe_poses(
            keyframes,
            edges_local,
            estimate_scale=estimate_scale and graph_factor_mode,
        )
        unique_poses = apply_pose_graph_corrections(
            initial_poses, keyframes, optimized
        )
        poses = [unique_poses[index] for index in original_to_pose]
        return poses

    if deskew_mode == 'full_se2':
        frames, se2_stats = full_se2_deskew_from_poses(
            frames, metadata, poses
        )
        rotation_deskew_stats.update(se2_stats)
        if se2_stats.get('se2_deskewed_points'):
            print(
                '  full SE2 deskew: '
                f'{se2_stats["se2_deskewed_points"]} points in '
                f'{se2_stats["se2_deskewed_frames"]} frames'
            )
            poses = _reestimate_after_deskew()
    elif deskew_mode in ('wheel+yaw', 'wheel+yaw_rate'):
        require_wheel = deskew_mode == 'wheel+yaw'
        max_ang = (
            math.radians(5.0)
            if deskew_mode == 'wheel+yaw_rate'
            else POINT_DESKEW_MAX_ANGLE
        )
        frames, rotation_deskew_stats = rotational_deskew_from_poses(
            frames,
            metadata,
            poses,
            encoder_samples=encoder_samples,
            require_wheel_motion=require_wheel,
            max_angle=max_ang,
        )
        if rotation_deskew_stats['rotational_deskewed_points']:
            print(
                '  second-pass angular deskew: '
                f'{rotation_deskew_stats["rotational_deskewed_points"]} '
                f'points in '
                f'{rotation_deskew_stats["rotational_deskewed_frames"]} '
                'frames'
            )
            poses = _reestimate_after_deskew()
    graph_at = time.perf_counter()
    refine_stats = {'refined_keyframes': 0, 'tried': 0}
    if (
        map_builder != 'occupancy_raytrace'
        and keyframes
        and not long_run_sensor_guard
        and (factor_mode or not get_wheel_se2())
        and len(frames) == len(poses)
    ):
        draft_points, _, _, _ = build_global_map_filtered(
            frames,
            poses,
            encoders=encoders,
            skip_turn_wz=skip_turn_wz,
            min_hits=map_min_hits,
            frame_times=[load_frame_timestamp(data) for data in metadata],
            cell_m=cell_m,
            min_component_cells=min_component_cells,
        )
        if len(draft_points) >= 200:
            poses, refine_stats_icp = refine_poses_to_map(
                frames,
                poses,
                keyframes,
                draft_points,
                cell_m=cell_m,
                max_xy=0.14,
                max_th=math.radians(4.0),
                min_overlap=0.36,
                max_dist=0.38,
                min_error=0.16,
            )
            refine_stats.update(refine_stats_icp)
            if refine_stats.get('refined_keyframes'):
                draft2, _, _, _ = build_global_map_filtered(
                    frames,
                    poses,
                    encoders=encoders,
                    skip_turn_wz=skip_turn_wz,
                    min_hits=map_min_hits,
                    frame_times=[
                        load_frame_timestamp(data) for data in metadata
                    ],
                    cell_m=cell_m,
                    min_component_cells=min_component_cells,
                )
                poses, refine2 = refine_poses_to_map(
                    frames,
                    poses,
                    keyframes,
                    draft2,
                    cell_m=cell_m,
                    max_xy=0.10,
                    max_th=math.radians(3.0),
                    min_overlap=0.40,
                    max_dist=0.30,
                    min_error=0.12,
                )
                refine_stats['round2'] = refine2.get('refined_keyframes', 0)

    if map_builder == 'occupancy_raytrace':
        from map_builders import build_occupancy_raytrace
        map_points, occ_stats = build_occupancy_raytrace(
            frames, poses, cell_m=cell_m, frame_stride=1, hit_stride=2
        )
        transformed_count = int(sum(len(frame) for frame in frames))
        raw_cells = int(occ_stats.get('raw_cells', occ_stats.get('occupied_cells', len(map_points))))
    else:
        map_points, transformed_count, raw_cells, map_filter_stats = (
            build_global_map_filtered(
                frames,
                poses,
                encoders=encoders,
                skip_turn_wz=skip_turn_wz,
                min_hits=map_min_hits,
                frame_times=[load_frame_timestamp(data) for data in metadata],
                cell_m=cell_m,
                min_component_cells=min_component_cells,
            )
        )
        if map_filter_stats.get('skipped_turn_frames'):
            print(
                f'  skipped {map_filter_stats["skipped_turn_frames"]} turn '
                f'frames (|wz|≥{skip_turn_wz}) for hit-grid'
            )
        if get_wheel_se2() and len(map_points) >= 200:
            map_points, manhattan_snap = snap_map_to_manhattan(
                map_points, cell_m=cell_m,
            )
            refine_stats['manhattan_snap'] = manhattan_snap
    map_at = time.perf_counter()

    pose_records = []
    keyframe_indices = {
        item.get('source_frame_index', pose_indices[item['frame_index']])
        for item in keyframes
    }
    for index, (pose, data) in enumerate(zip(poses, metadata)):
        encoder = load_encoder_record(data)
        record = {
            'ts': data.get('ts'),
            't': load_frame_timestamp(data),
            'frame': data.get('frame', index + 1),
            'x': float(pose[0]),
            'y': float(pose[1]),
            'orientation_dgr': float(math.degrees(pose[2])),
            'pose': [float(value) for value in pose],
            'keyframe': index in keyframe_indices,
            'n_points': int(len(frames[index])),
            'map_cameras': data.get('map_cameras', cameras),
        }
        if encoder is not None:
            record['encoder'] = {
                'enc_t': encoder['enc_t_raw'],
                'enc_unix': encoder['enc_t'],
                'enc_v': encoder['enc_v'],
                'enc_dgr': encoder['enc_dgr'],
            }
        synced = encoders[index]
        if use_encoder and synced is not None:
            record['synced_encoder'] = {
                'frame_t': synced['frame_t'],
                'enc_v': synced['enc_v'],
                'enc_dgr': synced['enc_dgr'],
                'nearest_sample_dt_s': synced['sync_error_s'],
            }
        pose_records.append(record)

    print(
        f'global map: transformed ALL {transformed_count} points from '
        f'{len(frames)} frames; {raw_cells} raw cells -> '
        f'{len(map_points)} stable cells'
    )
    first_pose = poses[0]
    last_pose = poses[-1]
    stats = {
        'mode': (
            'factor'
            if factor_mode and use_encoder
            else (
                'lidar+encoder'
                if use_encoder and encoder_samples
                else 'lidar_only'
            )
        ),
        'map_cameras': cameras,
        'up_body_x': None if up_body_x is None else float(up_body_x),
        'factor_mode': bool(factor_mode),
        'factor_requested': bool(requested_factor_mode),
        'long_run_sensor_guard': bool(long_run_sensor_guard),
        'submap_mode': bool(submap_mode),
        **submap_stats,
        'factor_constraint_profile': factor_constraint_profile,
        's_v': float(scale_stats.get('s_v', 1.0)),
        'estimate_scale': bool(scale_stats.get('estimate_scale', False)),
        'skip_lidar_sequential': bool(skip_lidar_sequential),
        'skip_loop_closures': bool(skip_loop_closures),
        'frontend_icp': frontend_icp,
        'lidar_seq_target': lidar_seq_target,
        'lidar_odom_dt_s': float(dt_stats.get('best_dt_s') or 0.0),
        'lidar_odom_dt_cost': dt_stats.get('best_cost'),
        'lidar_odom_dt_half_width_s': dt_stats.get('half_width_s'),
        'lidar_seq_edges': graph_stats.get('lidar_seq_edges', 0),
        'lidar_seq_rmse_median': graph_stats.get('lidar_seq_rmse_median'),
        'lidar_seq_overlap_median': graph_stats.get('lidar_seq_overlap_median'),
        'lidar_seq_across_over_along_median': graph_stats.get(
            'lidar_seq_across_over_along_median'
        ),
        'loop_rmse_median': graph_stats.get('loop_rmse_median'),
        'submap_keyframes': graph_stats.get('submap_keyframes'),
        'frontend_hardening': graph_stats.get('frontend_hardening', False),
        'seq_max_rmse': graph_stats.get('seq_max_rmse'),
        'loop_best_of_k': graph_stats.get('loop_best_of_k'),
        'encoder_frames': len(encoder_samples) if use_encoder else 0,
        'n_frames': len(frames),
        'pose_estimation_frames': len(pose_frames),
        'duplicate_frames_skipped': duplicate_frames,
        'stable_cells': len(map_points),
        'raw_cells': raw_cells,
        'transformed_points': transformed_count,
        'pose_graph_cost': graph_cost,
        'wheel_edges': graph_stats['wheel_edges'],
        'loop_closures': graph_stats['loop_closures'],
        'scan_to_map_refined': refine_stats.get('refined_keyframes', 0),
        'scan_to_map_round2': refine_stats.get('round2', 0),
        'manhattan_snap': refine_stats.get('manhattan_snap') or {},
        'keyframes': graph_stats['keyframes'],
        'encoder_unique_samples': sync_stats['unique_samples'],
        'encoder_synced_frames': (
            sync_stats['synced_frames'] if use_encoder else 0
        ),
        'encoder_rejected_wide_bracket': (
            sync_stats['rejected_wide_bracket'] if use_encoder else 0
        ),
        'encoder_median_sync_error_s': (
            sync_stats['median_sync_error_s'] if use_encoder else None
        ),
        'encoder_max_sync_error_s': (
            sync_stats['max_sync_error_s'] if use_encoder else None
        ),
        'encoder_max_bracket_span_s': (
            sync_stats['max_bracket_span_s'] if use_encoder else None
        ),
        'deskewed_frames': deskew_stats.get('deskewed_frames', 0),
        'deskewed_points': deskew_stats.get('deskewed_points', 0),
        'deskew_max_correction_m': deskew_stats.get(
            'deskew_max_correction_m', 0.0
        ),
        'deskew_mode': deskew_mode,
        'map_builder': map_builder,
        'cell_m': cell_m,
        'keyframe_xy': keyframe_xy,
        'keyframe_th_deg': math.degrees(keyframe_th),
        'adaptive_turn_keyframes': adaptive_turn_keyframes,
        'rotational_deskewed_frames': (
            rotation_deskew_stats.get('rotational_deskewed_frames', 0)
        ),
        'rotational_deskewed_points': (
            rotation_deskew_stats.get('rotational_deskewed_points', 0)
        ),
        'deskew_max_angle_deg': (
            rotation_deskew_stats.get('deskew_max_angle_deg', 0.0)
        ),
        'se2_deskewed_frames': rotation_deskew_stats.get(
            'se2_deskewed_frames', 0
        ),
        'se2_deskewed_points': rotation_deskew_stats.get(
            'se2_deskewed_points', 0
        ),
        'loop_error_m': math.hypot(
            last_pose[0] - first_pose[0], last_pose[1] - first_pose[1]
        ),
        'loop_error_deg': abs(
            math.degrees(normalize_angle(last_pose[2] - first_pose[2]))
        ),
        'timing_s': {
            'load_and_sync': loaded_at - total_started,
            'trajectory': trajectory_at - loaded_at,
            'pose_graph': graph_at - trajectory_at,
            'global_map': map_at - graph_at,
            'total': map_at - total_started,
        },
    }
    if return_internals:
        return map_points, pose_records, files, stats, frames, poses
    return map_points, pose_records, files, stats

def collect_http_run(
    output_dir, duration, cameras=None, base_url=None, poll_hz=20.0,
    encoder_poll_hz=20.0, build_map=True, sample_hz=None,
):
    if base_url:
        set_base_url(base_url)
    else:
        resolve_base_url()
    local_dir = Path(output_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / 'horizon').mkdir(parents=True, exist_ok=True)
    (local_dir / 'lidar.config').write_text(json.dumps({
        'format': 'lidar.config.v1',
        'lidar_type': 'rplidar',
        'sensor': 'RPLIDAR',
        # combine_camera_clouds() stores root horizon points in body frame.
        'points_frame': 'tracking',
        'coordinate_system': {
            'frame': 'rotation_axis', 'units': 'm',
            'x': 'forward', 'y': 'left', 'z': 'up',
        },
        'position_m': {'x': 1.015, 'y': 0.0, 'z': 1.645},
        'orientation_deg': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        'sensors': {
            'left': {'x': 1.710, 'y': 0.385, 'z': 0.135, 'yaw_deg': 45.0},
            'right': {'x': 1.710, 'y': -0.385, 'z': 0.135, 'yaw_deg': -45.0},
            'up': {'x': 1.015, 'y': 0.0, 'z': 1.645, 'yaw_deg': 0.0},
        },
        'source': 'real_robot_mount_geometry',
    }, indent=2) + '\n', encoding='utf-8')
    frame_index = 0
    poll_hz = max(float(poll_hz), 0.1)
    interval = 1.0 / poll_hz
    started = time.monotonic()
    deadline = started + duration if duration > 0 else float('inf')
    cameras = normalize_camera_list(cameras or MAP_CAMERAS)
    camera_frame_indices = {camera: 0 for camera in cameras}
    last_camera_stamps = {}
    last_saved_stamp = None
    duplicate_polls = 0
    lidar_clock_offset_s = None
    lidar_timestamp_domain = 'unix'
    encoder_poll_hz = max(float(encoder_poll_hz), 0.1)
    encoder_stats = {'polls': 0, 'unique': 0, 'duplicates': 0, 'errors': 0}
    encoder_stop = threading.Event()

    def _poll_encoder():
        # requests.Session is not guaranteed to be thread-safe; keep the
        # odometry transport independent from /raw_points polling.
        encoder_session = requests.Session()
        encoder_session.verify = False
        encoder_session.trust_env = False
        encoder_interval = 1.0 / encoder_poll_hz
        last_stamp = None
        while not encoder_stop.is_set():
            poll_started = time.monotonic()
            encoder = fetch_encoder_log(http_session=encoder_session)
            encoder_stats['polls'] += 1
            if encoder is None:
                encoder_stats['errors'] += 1
            elif encoder['enc_t_raw'] != last_stamp:
                last_stamp = encoder['enc_t_raw']
                append_encoder_sample(local_dir, encoder)
                encoder_stats['unique'] += 1
            else:
                encoder_stats['duplicates'] += 1
            remaining = encoder_interval - (time.monotonic() - poll_started)
            encoder_stop.wait(max(0.0, remaining))
        encoder_session.close()

    writer = AsyncNpzWriter(max_queue=256)
    for camera in cameras:
        (local_dir / camera).mkdir(parents=True, exist_ok=True)
    print(
        f'recording {duration}s at HTTP poll {poll_hz:g}Hz -> {local_dir}\n'
        f'robot={_source_io.BASE}; npz=horizon,left,right,up/*.npz; '
        f'map cameras={cameras}; encoder poll={encoder_poll_hz:g}Hz'
    )
    encoder_thread = threading.Thread(
        target=_poll_encoder, name='check-logs-poller', daemon=True,
    )
    encoder_thread.start()

    try:
        while time.monotonic() < deadline:
            frame_started = time.monotonic()
            local_points, frame_t, point_times, camera_clouds = fetch_raw_points(
                cameras=cameras, record_all=True
            )
            # /raw_points on some robot images uses boot-relative seconds;
            # align that domain with /check_logs before writing/synchronizing.
            previous_offset = lidar_clock_offset_s
            lidar_clock_offset_s, lidar_timestamp_domain = (
                normalize_http_lidar_timestamps(
                    camera_clouds,
                    host_time_s=time.time(),
                    offset_s=lidar_clock_offset_s,
                )
            )
            if camera_clouds and (
                lidar_clock_offset_s != previous_offset
                or lidar_timestamp_domain == 'monotonic_host_anchored'
            ):
                local_points, frame_t, point_times, _used = combine_camera_clouds(
                    camera_clouds, cameras=cameras
                )
            changed_cameras = []
            for camera in cameras:
                cloud = camera_clouds.get(camera)
                if cloud is None:
                    continue
                stamp = cloud.get('t')
                if stamp is None or stamp == last_camera_stamps.get(camera):
                    continue
                last_camera_stamps[camera] = stamp
                changed_cameras.append(camera)
                camera_frame_indices[camera] += 1
                save_horizon_npz(
                    local_dir,
                    camera_frame_indices[camera],
                    cloud['local_points'],
                    frame_t=stamp,
                    point_times=cloud.get('point_times'),
                    writer=writer,
                    subdir=camera,
                )

            if changed_cameras and (len(local_points) or camera_clouds):
                if (
                    sample_hz is not None and float(sample_hz) > 0.0
                    and last_saved_stamp is not None
                    and frame_t - last_saved_stamp < 1.0 / float(sample_hz) - 1e-9
                ):
                    changed_cameras = []
                    continue
                last_saved_stamp = frame_t
                frame_index += 1
                if len(local_points) == 0 and camera_clouds:
                    local_points, frame_t, point_times, _used = (
                        combine_camera_clouds(
                            camera_clouds, cameras=list(camera_clouds.keys())
                        )
                    )
                save_horizon_npz(
                    local_dir,
                    frame_index,
                    local_points,
                    frame_t=frame_t,
                    point_times=point_times,
                    writer=writer,
                )
            else:
                duplicate_polls += 1
            if frame_index and frame_index % 20 == 0:
                elapsed = time.monotonic() - started
                print(
                    f'captured={frame_index} fps={frame_index / elapsed:.1f} '
                    f'npz_saved={writer._saved} drop={writer._dropped} '
                    f'duplicate_polls={duplicate_polls}'
                )
            remaining = interval - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        encoder_stop.set()
        encoder_thread.join(timeout=2.0)
        writer.close()

    elapsed = time.monotonic() - started
    print(
        f'captured {frame_index} frames in {elapsed:.1f}s '
        f'({frame_index / elapsed:.1f} unique fps); '
        f'duplicate_polls={duplicate_polls}; per_camera={camera_frame_indices}; '
        f'encoder={encoder_stats}'
    )
    summary = {
        'duration_s': elapsed,
        'lidar_poll_hz': poll_hz,
        'lidar_unique_frames': frame_index,
        'lidar_unique_hz': frame_index / elapsed if elapsed else 0.0,
        'lidar_duplicate_polls': duplicate_polls,
        'lidar_timestamp_domain': lidar_timestamp_domain,
        'lidar_clock_offset_s': lidar_clock_offset_s,
        'lidar_frames_per_camera': camera_frame_indices,
        'encoder_poll_hz': encoder_poll_hz,
        'encoder': encoder_stats,
        'encoder_unique_hz': (
            encoder_stats['unique'] / elapsed if elapsed else 0.0
        ),
    }
    (local_dir / 'recording_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )
    if not build_map:
        print(f'calibration recording saved: {local_dir}')
        return local_dir
    points, poses, _, stats = sew_all_local_frames(
        local_dir, cameras=cameras
    )
    used_encoder = bool(stats.get('encoder_frames'))
    out_json, out_png = save_global_map(
        points, poses, local_dir.name, local_dir, duration, used_encoder, source='real'
    )
    print(f'saved: {out_json}\nsaved: {out_png}')
    print(f'encoder log: {encoder_log_path(local_dir)}')
    print(f'map cameras used: {stats.get("map_cameras")}')
    return local_dir


def run(duration):
    run_id = f"run_{datetime.now().strftime('%H%M%S')}"
    local_dir, _results_dir = make_run_directories('real', run_name=run_id)
    return collect_http_run(local_dir, duration, cameras=MAP_CAMERAS)


sew_frames = motion_from_scans
match_frames = motion_from_scans


def cell_key(x, y):
    return (
        round(x / MAP_CELL) * MAP_CELL,
        round(y / MAP_CELL) * MAP_CELL,
    )


if __name__ == '__main__':
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    run(seconds)
