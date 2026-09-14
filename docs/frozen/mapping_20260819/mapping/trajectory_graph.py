"""Trajectory estimation and pose-graph optimization."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .config import (
    FACTOR_ESTIMATE_SCALE, FACTOR_LIDAR_BASE_WEIGHT, FACTOR_LOOP_BASE_WEIGHT,
    FACTOR_ODOM_SIGMA_XY, FACTOR_ODOM_SIGMA_YAW, FACTOR_SCALE_PRIOR_SIGMA,
    FRONTEND_SCAN_MAX_RMSE, FRONTEND_SCAN_MIN_OVERLAP,
    FRONTEND_SUBMAP_MAX_DIST, FRONTEND_WHEEL_AGREE_TH, FRONTEND_WHEEL_AGREE_XY,
    KEYFRAME_TH, KEYFRAME_XY, LOOP_BEST_OF_K, LOOP_FRAME_GAP,
    LOOP_FRAME_GAP_AGGRO, LOOP_MAX_CORR_TH, LOOP_MAX_CORR_TH_AGGRO,
    LOOP_MAX_CORR_XY, LOOP_MAX_CORR_XY_AGGRO, LOOP_MAX_ERROR,
    LOOP_MAX_ERROR_AGGRO, LOOP_MIN_OVERLAP, LOOP_MIN_OVERLAP_AGGRO,
    LOOP_RADIUS, LOOP_RADIUS_AGGRO, MAP_CELL, POSE_GRAPH_SPARSE_JACOBIAN,
    SEQ_MAX_RMSE, SEQ_MIN_OVERLAP, SEQ_REQUIRE_WHEEL_AGREE, SUBMAP_KEYFRAMES,
    TURN_KEYFRAME_TH, TURN_KEYFRAME_XY, WHEEL_EDGE_WEIGHT,
)
from .encoder_sync import fuse_motion, get_wheel_se2, motion_from_encoders
from .geometry_icp import (
    compose, estimate_icp_information, icp_to_tree, inverse_pose,
    local_to_global, longest_line_in_scan, motion_from_scans, normalize_angle,
    pose_changed, relative_pose, robust_icp, voxel_downsample,
)


def _wheel_has_yaw(wheel_delta):
    return wheel_delta is not None and abs(wheel_delta[2]) > 1e-6


def _encoder_reports_wz(encoder):
    return get_wheel_se2() and encoder is not None and encoder.get('enc_wz') is not None


def _scan_agrees_with_wheel(scan_delta, wheel_delta, max_xy=None, max_th=None):
    if wheel_delta is None or scan_delta is None:
        return False
    max_xy = FRONTEND_WHEEL_AGREE_XY if max_xy is None else float(max_xy)
    max_th = FRONTEND_WHEEL_AGREE_TH if max_th is None else float(max_th)
    disagree = relative_pose(wheel_delta, scan_delta)
    return (
        math.hypot(disagree[0], disagree[1]) <= max_xy
        and abs(disagree[2]) <= max_th
    )


def _scan_delta_reliable(scan_error, scan_overlap, scan_delta, wheel_delta):
    """Accept wheel-seeded scan ICP only if quality + (optional) wheel agree."""
    if not (np.isfinite(scan_error) and scan_overlap >= FRONTEND_SCAN_MIN_OVERLAP):
        return False
    if scan_error > FRONTEND_SCAN_MAX_RMSE:
        return False
    if wheel_delta is None:
        return True
    return _scan_agrees_with_wheel(scan_delta, wheel_delta)


def _unpack_edge(edge):
    """Normalize edge to (i, j, measured, weight, kind)."""
    if len(edge) >= 5:
        return edge[0], edge[1], edge[2], edge[3], edge[4]
    return edge[0], edge[1], edge[2], edge[3], 'generic'


def _residual_weights(weight):
    """Map scalar / (w_xy, w_yaw) / (wx, wy, wyaw) → (wx, wy, wyaw)."""
    if isinstance(weight, tuple):
        if len(weight) >= 3:
            return float(weight[0]), float(weight[1]), float(weight[2])
        if len(weight) == 2:
            return float(weight[0]), float(weight[0]), float(weight[1])
    w = float(weight)
    return w, w, w


def make_submap(keyframes, cell_m=None, n=None):
    """Voxelized local map from the last ``n`` keyframes (default SUBMAP_KEYFRAMES)."""
    if not keyframes:
        return np.empty((0, 2), dtype=np.float64)
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    n_keep = SUBMAP_KEYFRAMES if n is None else max(1, int(n))
    clouds = [
        local_to_global(item['points'], item['pose'])
        for item in keyframes[-n_keep:]
        if len(item.get('points', [])) > 0
    ]
    if not clouds:
        return np.empty((0, 2), dtype=np.float64)
    return voxel_downsample(np.vstack(clouds), cell_m)


def estimate_initial_trajectory(
    frames,
    encoders=None,
    use_encoder=True,
    source_indices=None,
    skip_duplicate_pair_icp=True,
    keyframe_xy=None,
    keyframe_th=None,
    adaptive_turn_keyframes=False,
    icp_max_xy=0.35,
    icp_max_th=None,
    icp_min_overlap=0.42,
    cell_m=None,
    factor_mode=False,
    frontend_icp='auto',
    submap_keyframes=None,
):
    """Estimate frame poses.

    frontend_icp:
      auto   — legacy: wheel SE2 only when get_wheel_se2() and not factor_mode;
               else scan-to-scan then scan-to-submap refine
      wheel  — force wheel-only frame deltas (no scan ICP)
      scan   — consecutive scan-to-scan ICP (seeded by wheel when available)
      submap — scan-to-submap refine against last N keyframes
    """
    from .config import SUBMAP_KEYFRAMES

    encoders = encoders or [None] * len(frames)
    source_indices = source_indices or list(range(len(frames)))
    if not use_encoder:
        encoders = [None] * len(frames)
    keyframe_xy = KEYFRAME_XY if keyframe_xy is None else keyframe_xy
    keyframe_th = KEYFRAME_TH if keyframe_th is None else keyframe_th
    icp_max_th = math.radians(8.0) if icp_max_th is None else icp_max_th
    frontend_icp = str(frontend_icp or 'auto').lower()
    submap_n = (
        SUBMAP_KEYFRAMES if submap_keyframes is None else int(submap_keyframes)
    )
    wheel_poses = [[0.0, 0.0, 0.0]]
    wheel_valid_prefix = [0]
    poses = [[0.0, 0.0, 0.0]]
    keyframes = [{
        'frame_index': 0,
        'source_frame_index': source_indices[0],
        'points': frames[0],
        'pose': poses[0].copy(),
        'encoder': encoders[0],
        'wheel_pose': wheel_poses[0],
        'wheel_valid_prefix': wheel_valid_prefix[0],
    }]
    previous = frames[0]
    previous_encoder = encoders[0]

    for index, current in enumerate(frames[1:], start=1):
        current_encoder = encoders[index] if index < len(encoders) else None
        wheel_delta = motion_from_encoders(previous_encoder, current_encoder)
        exact_repeat = (
            skip_duplicate_pair_icp
            and previous.shape == current.shape
            and np.array_equal(previous, current)
        )

        # factor_mode: wheel is only an ICP seed / soft prior, never the sole delta
        # unless frontend_icp='wheel' (diagnostic ablation).
        force_wheel = frontend_icp == 'wheel'
        use_legacy_wheel = (
            frontend_icp == 'auto'
            and wheel_delta is not None
            and get_wheel_se2()
            and not factor_mode
        )
        if exact_repeat:
            scan_delta, scan_error, scan_overlap = (
                [0.0, 0.0, 0.0], 0.0, 1.0
            )
            scan_reliable = True
            delta = scan_delta
        elif force_wheel or use_legacy_wheel:
            if wheel_delta is None:
                scan_delta, scan_error, scan_overlap = motion_from_scans(
                    previous, current, return_quality=True,
                )
                scan_reliable = _scan_delta_reliable(
                    scan_error, scan_overlap, scan_delta, None,
                )
                delta = scan_delta if scan_reliable else [0.0, 0.0, 0.0]
            else:
                delta = list(wheel_delta)
                scan_delta = delta
                scan_reliable = False
                scan_error, scan_overlap = float('inf'), 0.0
        elif wheel_delta is not None:
            scan_delta, scan_error, scan_overlap = motion_from_scans(
                previous,
                current,
                initial_pose=wheel_delta,
                return_quality=True,
            )
            scan_reliable = _scan_delta_reliable(
                scan_error, scan_overlap, scan_delta, wheel_delta,
            )
            if factor_mode:
                delta = scan_delta if scan_reliable else list(wheel_delta)
            else:
                delta = scan_delta if scan_reliable else fuse_motion(
                    wheel_delta, scan_delta
                )
        else:
            scan_delta, scan_error, scan_overlap = motion_from_scans(
                previous,
                current,
                return_quality=True,
            )
            scan_reliable = _scan_delta_reliable(
                scan_error, scan_overlap, scan_delta, None,
            )
            delta = scan_delta if scan_reliable else fuse_motion(
                wheel_delta, scan_delta
            )

        if wheel_delta is None:
            wheel_poses.append(wheel_poses[-1].copy())
            wheel_valid_prefix.append(wheel_valid_prefix[-1])
        else:
            wheel_increment = [
                wheel_delta[0],
                wheel_delta[1],
                wheel_delta[2] if abs(wheel_delta[2]) > 1e-6 else (
                    scan_delta[2] if scan_reliable else delta[2]
                ),
            ]
            wheel_poses.append(compose(wheel_poses[-1], wheel_increment))
            wheel_valid_prefix.append(wheel_valid_prefix[-1] + 1)

        skip_submap = (
            force_wheel
            or frontend_icp == 'scan'
            or (
                frontend_icp == 'auto'
                and get_wheel_se2()
                and not factor_mode
            )
        )
        if skip_submap or exact_repeat:
            pose = compose(poses[-1], delta)
            poses.append(pose)
        else:
            guess = compose(poses[-1], delta)
            fallback = guess
            # Pass n explicitly so SUBMAP_KEYFRAMES / submap_keyframes is honoured.
            sub_kfs = keyframes[-max(1, submap_n):]
            submap = make_submap(sub_kfs, cell_m=cell_m, n=len(sub_kfs))
            refined, error, overlap = robust_icp(
                current,
                submap,
                initial_pose=guess,
                max_dist=FRONTEND_SUBMAP_MAX_DIST,
            )
            correction = relative_pose(guess, refined)
            correction_ok = (
                np.isfinite(error)
                and error <= FRONTEND_SCAN_MAX_RMSE
                and overlap >= icp_min_overlap
                and math.hypot(correction[0], correction[1]) <= icp_max_xy
                and abs(correction[2]) <= icp_max_th
            )
            pose = refined if correction_ok else fallback
            poses.append(pose)

        yaw_step = abs(normalize_angle(pose[2] - poses[index - 1][2]))
        use_xy = keyframe_xy
        use_th = keyframe_th
        if adaptive_turn_keyframes and yaw_step >= math.radians(0.5):
            use_xy = min(use_xy, TURN_KEYFRAME_XY)
            use_th = min(use_th, TURN_KEYFRAME_TH)
        if pose_changed(
            keyframes[-1]['pose'], pose,
            min_xy=use_xy, min_angle=use_th,
        ) and len(current) >= 80:
            keyframes.append({
                'frame_index': index,
                'source_frame_index': source_indices[index],
                'points': current,
                'pose': pose.copy(),
                'encoder': current_encoder,
                'wheel_pose': wheel_poses[index],
                'wheel_valid_prefix': wheel_valid_prefix[index],
            })
        previous = current
        previous_encoder = current_encoder

        if (index + 1) % 200 == 0:
            print(
                f'  trajectory {index + 1}/{len(frames)} '
                f'keyframes={len(keyframes)}'
            )
    return poses, keyframes


def build_pose_graph_edges(
    keyframes,
    use_encoder=True,
    loop_aggro=False,
    factor_mode=False,
    estimate_scale=None,
    skip_lidar_sequential=False,
    skip_loop_closures=False,
    lidar_seq_target='scan',
    submap_keyframes=None,
):
    """Build relative SE(2) constraints between keyframes.

    factor_mode: always add soft wheel + sequential lidar (no wheel-only
    continue). Edges are 5-tuples (i, j, measured, weight, kind).

    Diagnostic knobs (defaults preserve current factor behaviour):
      skip_lidar_sequential — omit sequential lidar edges
      skip_loop_closures — omit loop edges (loops never use /odom)
      lidar_seq_target — 'scan' (prev keyframe) or 'submap' (last N keyframes)
    """
    from .config import SUBMAP_KEYFRAMES

    if estimate_scale is None:
        estimate_scale = FACTOR_ESTIMATE_SCALE if factor_mode else False
    skip_loop_closures = bool(skip_loop_closures)
    lidar_seq_target = str(lidar_seq_target or 'scan').lower()
    submap_n = (
        SUBMAP_KEYFRAMES if submap_keyframes is None else int(submap_keyframes)
    )
    if loop_aggro:
        loop_radius = LOOP_RADIUS_AGGRO
        loop_frame_gap = LOOP_FRAME_GAP_AGGRO
        loop_max_error = LOOP_MAX_ERROR_AGGRO
        loop_min_overlap = LOOP_MIN_OVERLAP_AGGRO
        loop_max_corr_xy = LOOP_MAX_CORR_XY_AGGRO
        loop_max_corr_th = LOOP_MAX_CORR_TH_AGGRO
    elif get_wheel_se2():
        loop_radius = 2.2
        loop_frame_gap = 80
        loop_max_error = 0.16
        loop_min_overlap = 0.32
        loop_max_corr_xy = 0.45
        loop_max_corr_th = math.radians(8.0)
    else:
        loop_radius = LOOP_RADIUS
        loop_frame_gap = LOOP_FRAME_GAP
        loop_max_error = LOOP_MAX_ERROR
        loop_min_overlap = LOOP_MIN_OVERLAP
        loop_max_corr_xy = LOOP_MAX_CORR_XY
        loop_max_corr_th = LOOP_MAX_CORR_TH
    lookback = len(keyframes)
    edges = []
    wheel_edges = 0
    lidar_seq_edges = 0
    lidar_seq_rmse = []
    lidar_seq_overlap = []
    lidar_seq_aniso = []
    loop_rmse = []
    odom_sigma_xy = max(1e-3, float(FACTOR_ODOM_SIGMA_XY))
    odom_sigma_yaw = max(1e-4, float(FACTOR_ODOM_SIGMA_YAW))
    odom_w_xy = 1.0 / odom_sigma_xy
    odom_w_yaw = 1.0 / odom_sigma_yaw

    for index in range(1, len(keyframes)):
        previous = keyframes[index - 1]
        current = keyframes[index]
        initial = relative_pose(previous['pose'], current['pose'])
        frame_gap = current['frame_index'] - previous['frame_index']
        valid_intervals = (
            current.get('wheel_valid_prefix', 0)
            - previous.get('wheel_valid_prefix', 0)
        )
        coverage = valid_intervals / max(1, frame_gap)
        wheel_delta = relative_pose(
            previous.get('wheel_pose', [0.0, 0.0, 0.0]),
            current.get('wheel_pose', [0.0, 0.0, 0.0]),
        )
        wheel_ok = (
            use_encoder
            and coverage >= 0.80
            and wheel_delta != [0.0, 0.0, 0.0]
        )

        if factor_mode:
            # Soft wheel prior (scaled by s_v in the optimizer).
            if wheel_ok:
                cov_scale = max(0.45, float(coverage))
                edges.append((
                    index - 1,
                    index,
                    list(wheel_delta),
                    (
                        odom_w_xy * cov_scale,
                        odom_w_xy * cov_scale,
                        odom_w_yaw * cov_scale if abs(wheel_delta[2]) > 1e-6 else 0.0,
                    ),
                    'wheel',
                ))
                wheel_edges += 1

            if not skip_lidar_sequential:
                seed = wheel_delta if wheel_ok else initial
                if lidar_seq_target == 'submap' and index >= 1:
                    sub_kfs = keyframes[max(0, index - submap_n):index]
                    target_chunks = []
                    for kf in sub_kfs:
                        rel = relative_pose(previous['pose'], kf['pose'])
                        pts = np.asarray(kf['points'], float)
                        if len(pts) == 0:
                            continue
                        c, s = math.cos(rel[2]), math.sin(rel[2])
                        x = rel[0] + c * pts[:, 0] - s * pts[:, 1]
                        y = rel[1] + s * pts[:, 0] + c * pts[:, 1]
                        target_chunks.append(np.column_stack([x, y]))
                    target_cloud = (
                        np.vstack(target_chunks)
                        if target_chunks
                        else previous['points']
                    )
                else:
                    target_cloud = previous['points']
                measurement, error, overlap = robust_icp(
                    current['points'],
                    target_cloud,
                    initial_pose=seed,
                )
                sequential_reliable = (
                    np.isfinite(error)
                    and overlap >= SEQ_MIN_OVERLAP
                    and error <= SEQ_MAX_RMSE
                )
                if (
                    sequential_reliable
                    and SEQ_REQUIRE_WHEEL_AGREE
                    and wheel_ok
                ):
                    sequential_reliable = _scan_agrees_with_wheel(
                        measurement, wheel_delta,
                    )
                if sequential_reliable:
                    info, details = estimate_icp_information(
                        target_cloud,
                        rmse=error,
                        overlap=overlap,
                        base_weight=FACTOR_LIDAR_BASE_WEIGHT,
                        return_details=True,
                    )
                    edges.append((
                        index - 1, index, measurement, info, 'lidar',
                    ))
                    lidar_seq_edges += 1
                    lidar_seq_rmse.append(float(error))
                    lidar_seq_overlap.append(float(overlap))
                    if details.get('across_over_along') is not None:
                        lidar_seq_aniso.append(float(details['across_over_along']))
                elif not wheel_ok:
                    edges.append((
                        index - 1, index, initial, 0.5, 'lidar',
                    ))
                    lidar_seq_edges += 1
            continue

        # ---- legacy paths (improved / odom / clouds) ----
        wheel_se2 = wheel_ok and get_wheel_se2()
        if wheel_se2:
            edge_weight = max(0.45, 0.6 * coverage)
            edges.append((
                index - 1, index, wheel_delta, (edge_weight, edge_weight),
            ))
            wheel_edges += 1
            continue

        measurement, error, overlap = robust_icp(
            current['points'], previous['points'], initial_pose=initial
        )
        sequential_reliable = np.isfinite(error) and overlap >= 0.38
        if sequential_reliable:
            edges.append((index - 1, index, measurement, 1.0))
            lidar_seq_edges += 1
        else:
            edges.append((index - 1, index, initial, 0.5))
            lidar_seq_edges += 1

        wheel_scan_disagreement = relative_pose(measurement, wheel_delta)
        wheel_agrees = (
            math.hypot(
                wheel_scan_disagreement[0], wheel_scan_disagreement[1]
            ) <= 0.30
            and abs(wheel_scan_disagreement[2]) <= math.radians(10.0)
        )
        if wheel_ok and wheel_agrees:
            edge_weight = WHEEL_EDGE_WEIGHT * coverage
            yaw_weight = edge_weight if abs(wheel_delta[2]) > 1e-6 else 0.0
            edges.append((
                index - 1,
                index,
                wheel_delta,
                (edge_weight, yaw_weight),
            ))
            wheel_edges += 1

    loop_edges = 0
    loop_max_dist = 0.55 if (loop_aggro or get_wheel_se2()) else 0.40
    if not skip_loop_closures:
        for current_index in range(len(keyframes)):
            current = keyframes[current_index]
            candidates = []
            for old_index in range(max(0, current_index - lookback), current_index):
                old = keyframes[old_index]
                if (
                    current.get('source_frame_index', current['frame_index'])
                    - old.get('source_frame_index', old['frame_index'])
                    < loop_frame_gap
                ):
                    continue
                distance = math.hypot(
                    current['pose'][0] - old['pose'][0],
                    current['pose'][1] - old['pose'][1],
                )
                if distance <= loop_radius:
                    if len(old.get('points', [])) < 80:
                        continue
                    candidates.append((distance, old_index))
            if len(current.get('points', [])) < 80:
                continue
            best = None  # (error, old_index, measurement, overlap)
            for _, old_index in sorted(candidates)[:max(1, int(LOOP_BEST_OF_K))]:
                old = keyframes[old_index]
                # Loop pose seed is from LiDAR trajectory only — never /odom.
                initial = relative_pose(old['pose'], current['pose'])
                measurement, error, overlap = robust_icp(
                    current['points'],
                    old['points'],
                    initial_pose=initial,
                    max_dist=loop_max_dist,
                )
                correction = relative_pose(initial, measurement)
                if not (
                    np.isfinite(error)
                    and error <= loop_max_error
                    and overlap >= loop_min_overlap
                    and math.hypot(correction[0], correction[1]) <= loop_max_corr_xy
                    and abs(correction[2]) <= loop_max_corr_th
                ):
                    continue
                if best is None or error < best[0]:
                    best = (float(error), old_index, measurement, float(overlap), old)
            if best is not None:
                error, old_index, measurement, overlap, old = best
                if factor_mode:
                    info = estimate_icp_information(
                        old['points'],
                        rmse=error,
                        overlap=overlap,
                        base_weight=FACTOR_LOOP_BASE_WEIGHT,
                    )
                    edges.append((
                        old_index, current_index, measurement, info, 'loop',
                    ))
                else:
                    edges.append(
                        (old_index, current_index, measurement, 5.0)
                    )
                loop_edges += 1
                loop_rmse.append(float(error))
    def _med(xs):
        return float(np.median(xs)) if xs else None

    print(
        f'  pose graph: {len(keyframes)} nodes, '
        f'{wheel_edges} wheel edges, {lidar_seq_edges} lidar seq, '
        f'{loop_edges} loop closures'
        + (' (factor)' if factor_mode else '')
        + (' (aggro loops)' if loop_aggro else '')
        + (' (no loops)' if skip_loop_closures else '')
        + (f' (seq={lidar_seq_target})' if factor_mode else '')
    )
    return edges, {
        'wheel_edges': wheel_edges,
        'lidar_seq_edges': lidar_seq_edges,
        'loop_closures': loop_edges,
        'keyframes': len(keyframes),
        'factor_mode': bool(factor_mode),
        'estimate_scale': bool(estimate_scale) if factor_mode else False,
        'skip_loop_closures': bool(skip_loop_closures),
        'lidar_seq_target': lidar_seq_target,
        'submap_keyframes': submap_n,
        'lidar_seq_rmse_median': _med(lidar_seq_rmse),
        'lidar_seq_overlap_median': _med(lidar_seq_overlap),
        'lidar_seq_across_over_along_median': _med(lidar_seq_aniso),
        'loop_rmse_median': _med(loop_rmse),
        'frontend_hardening': True,
        'seq_max_rmse': float(SEQ_MAX_RMSE),
        'loop_best_of_k': int(LOOP_BEST_OF_K),
    }


def optimize_keyframe_poses(
    keyframes,
    edges,
    estimate_scale=False,
    scale_prior_sigma=None,
):
    """Optimize keyframe SE(2) poses; optionally joint wheel scale s_v.

    Returns (poses, cost, extras) where extras includes s_v.
    """
    if len(keyframes) < 2:
        return [item['pose'] for item in keyframes], 0.0, {
            's_v': 1.0, 'estimate_scale': False,
        }
    scale_prior_sigma = (
        FACTOR_SCALE_PRIOR_SIGMA
        if scale_prior_sigma is None
        else float(scale_prior_sigma)
    )
    n_pose = len(keyframes) - 1
    pose_init = np.asarray(
        [value for item in keyframes[1:] for value in item['pose']],
        dtype=np.float64,
    )
    if estimate_scale:
        initial = np.concatenate([pose_init, np.asarray([1.0], dtype=np.float64)])
    else:
        initial = pose_init
    anchor = keyframes[0]['pose']

    def unpack_poses(values):
        pose_vals = values[: 3 * n_pose]
        result = [anchor]
        result.extend(pose_vals.reshape(-1, 3).tolist())
        return result

    def residual(values):
        poses = unpack_poses(values)
        s_v = float(values[-1]) if estimate_scale else 1.0
        errors = []
        for edge in edges:
            first, second, measured, weight, kind = _unpack_edge(edge)
            predicted = relative_pose(poses[first], poses[second])
            mx = float(measured[0])
            my = float(measured[1])
            mth = float(measured[2])
            if kind == 'wheel' and estimate_scale:
                mx *= s_v
                my *= s_v
            wx, wy, wyaw = _residual_weights(weight)
            errors.extend([
                wx * (predicted[0] - mx),
                wy * (predicted[1] - my),
                wyaw * 2.0 * normalize_angle(predicted[2] - mth),
            ])
        if estimate_scale:
            prior_w = 1.0 / max(1e-4, scale_prior_sigma)
            errors.append(prior_w * (s_v - 1.0))
        return np.asarray(errors, dtype=np.float64)

    n_edge_res = 3 * len(edges) + (1 if estimate_scale else 0)
    n_vars = 3 * n_pose + (1 if estimate_scale else 0)
    jacobian_sparsity = None
    if POSE_GRAPH_SPARSE_JACOBIAN:
        jacobian_sparsity = lil_matrix(
            (n_edge_res, n_vars), dtype=np.int8,
        )
        for edge_index, edge in enumerate(edges):
            first, second, _m, _w, kind = _unpack_edge(edge)
            rows = slice(3 * edge_index, 3 * edge_index + 3)
            if first > 0:
                columns = slice(3 * (first - 1), 3 * first)
                jacobian_sparsity[rows, columns] = 1
            if second > 0:
                columns = slice(3 * (second - 1), 3 * second)
                jacobian_sparsity[rows, columns] = 1
            if estimate_scale and kind == 'wheel':
                jacobian_sparsity[rows, -1] = 1
        if estimate_scale:
            jacobian_sparsity[-1, -1] = 1

    result = least_squares(
        residual,
        initial,
        jac_sparsity=jacobian_sparsity,
        loss='huber',
        f_scale=0.10,
        max_nfev=120,
        verbose=0,
    )
    optimized = unpack_poses(result.x)
    for pose in optimized:
        pose[2] = normalize_angle(pose[2])
    s_v = float(result.x[-1]) if estimate_scale else 1.0
    print(
        f'  pose graph optimized: cost {result.cost:.3f}, '
        f'iterations={result.nfev}'
        + (f', s_v={s_v:.5f}' if estimate_scale else '')
    )
    return optimized, float(result.cost), {
        's_v': s_v,
        'estimate_scale': bool(estimate_scale),
    }


def apply_pose_graph_corrections(frame_poses, keyframes, optimized):
    result = [None] * len(frame_poses)
    for key_index, keyframe in enumerate(keyframes):
        start = keyframe['frame_index']
        end = (
            keyframes[key_index + 1]['frame_index']
            if key_index + 1 < len(keyframes)
            else len(frame_poses)
        )
        correction = compose(
            optimized[key_index], inverse_pose(keyframe['pose'])
        )
        for frame_index in range(start, end):
            result[frame_index] = compose(correction, frame_poses[frame_index])
    return result


def _keyframe_frame_span(keyframes, key_index, n_poses):
    keyframe = keyframes[key_index]
    start = keyframe.get('source_frame_index', keyframe['frame_index'])
    if key_index + 1 < len(keyframes):
        end = keyframes[key_index + 1].get(
            'source_frame_index',
            keyframes[key_index + 1]['frame_index'],
        )
    else:
        end = n_poses
    return int(start), int(end)


def refine_poses_to_map(
    frames,
    poses,
    keyframes,
    map_points,
    cell_m=None,
    max_xy=0.28,
    max_th=None,
    min_overlap=0.38,
    max_dist=0.40,
    min_error=0.18,
    lateral_only=False,
):
    """Snap keyframe poses to a consensus hit-grid (scan-to-map)."""
    from scipy.spatial import cKDTree

    max_th = math.radians(6.0) if max_th is None else max_th
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    map_points = np.asarray(map_points, float)
    if len(map_points) < 200 or not keyframes:
        return poses, {'refined_keyframes': 0, 'tried': 0}

    target = voxel_downsample(map_points, cell=cell_m, max_points=14000)
    tree = cKDTree(target)
    result = [list(pose) for pose in poses]
    accepted = 0
    for key_index, keyframe in enumerate(keyframes):
        start, end = _keyframe_frame_span(keyframes, key_index, len(result))
        if start < 0 or start >= len(frames) or start >= len(result):
            continue
        cloud = frames[start]
        initial = list(result[start])
        if cloud is None or len(cloud) < 80:
            continue
        refined, error, overlap = icp_to_tree(
            cloud, tree, initial_pose=initial, max_dist=max_dist, cell_m=cell_m,
        )
        correction = relative_pose(initial, refined)
        if lateral_only:
            fitted = longest_line_in_scan(cloud, min_inliers=40, iters=70)
            if fitted is None or fitted[1] < 5.0:
                continue
            line_angle = fitted[0]
            normal_x = -math.sin(line_angle)
            normal_y = math.cos(line_angle)
            lateral = correction[0] * normal_x + correction[1] * normal_y
            correction = [
                lateral * normal_x,
                lateral * normal_y,
                correction[2],
            ]
            refined = compose(initial, correction)
        ok = (
            np.isfinite(error)
            and error <= min_error
            and overlap >= min_overlap
            and math.hypot(correction[0], correction[1]) <= max_xy
            and abs(correction[2]) <= max_th
        )
        if not ok:
            continue
        accepted += 1
        delta = compose(refined, inverse_pose(initial))
        for frame_index in range(start, min(end, len(result))):
            result[frame_index] = compose(delta, result[frame_index])
        keyframe['pose'] = list(result[start])
    print(
        f'  scan-to-map refine: {accepted}/{len(keyframes)} keyframes '
        f'accepted (map n={len(target)}'
        f'{", lateral-only" if lateral_only else ""})'
    )
    return result, {
        'refined_keyframes': accepted,
        'tried': len(keyframes),
        'map_points': int(len(target)),
        'lateral_only': bool(lateral_only),
    }
