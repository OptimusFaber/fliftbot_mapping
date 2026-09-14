"""SE(2) helpers and ICP scan matching."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    ICP_ITERS, ICP_MAX_DIST, ICP_MIN_INLIERS, ICP_POINTS, ICP_TRIM,
    KEYFRAME_TH, KEYFRAME_XY, MAP_CELL, MATCH_MIN_RANGE, MAX_FRAME_TH,
    MAX_FRAME_XY, STATIC_TH, STATIC_XY,
)

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def matching_points(points):
    if len(points) == 0:
        return points
    radius2 = np.sum(points * points, axis=1)
    return points[radius2 >= MATCH_MIN_RANGE * MATCH_MIN_RANGE]


def compose(first, second):
    """Compose SE(2) poses: world<-first<-second."""
    x1, y1, a1 = first
    x2, y2, a2 = second
    c, s = math.cos(a1), math.sin(a1)
    return [
        x1 + c * x2 - s * y2,
        y1 + s * x2 + c * y2,
        normalize_angle(a1 + a2),
    ]


def inverse_pose(pose):
    x, y, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    return [-c * x - s * y, s * x - c * y, -angle]


def relative_pose(first, second):
    """Pose of second in first coordinate system."""
    return compose(inverse_pose(first), second)


def local_to_global(local_points, pose):
    """Transform every robot-local point into the global coordinate system."""
    if len(local_points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    x, y, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local_points @ rotation.T + np.array([x, y], dtype=np.float64)


transform_points = local_to_global


def voxel_downsample(points, cell=MAP_CELL, max_points=ICP_POINTS):
    if len(points) == 0:
        return points
    cells = np.round(points / cell).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    result = points[np.sort(indices)]
    if max_points is not None and len(result) > max_points:
        step = max(1, len(result) // max_points)
        result = result[::step][:max_points]
    return result


def rigid_fit(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    return rotation, translation, angle


def robust_icp(
    source, target, initial_pose=None, max_dist=ICP_MAX_DIST,
    max_source=None, max_target=None, cell_m=None,
):
    """
    Align source points to target points.
    Returns source->target pose, RMSE and inlier ratio.
    """
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    max_source = ICP_POINTS if max_source is None else max_source
    max_target = ICP_POINTS if max_target is None else max_target
    source = voxel_downsample(
        matching_points(source), cell=cell_m, max_points=max_source
    )
    target = voxel_downsample(
        matching_points(target), cell=cell_m, max_points=max_target
    )
    if len(source) < ICP_MIN_INLIERS or len(target) < ICP_MIN_INLIERS:
        return list(initial_pose or (0.0, 0.0, 0.0)), float('inf'), 0.0

    pose = list(initial_pose or (0.0, 0.0, 0.0))
    tree = cKDTree(target)
    previous_error = float('inf')

    for _ in range(ICP_ITERS):
        transformed = local_to_global(source, pose)
        distances, indices = tree.query(transformed, k=1)
        valid = distances < max_dist
        if int(valid.sum()) < ICP_MIN_INLIERS:
            break

        valid_indices = np.flatnonzero(valid)
        keep_count = max(ICP_MIN_INLIERS, int(len(valid_indices) * ICP_TRIM))
        keep = valid_indices[np.argsort(distances[valid_indices])[:keep_count]]
        rotation, translation, angle = rigid_fit(
            transformed[keep], target[indices[keep]]
        )
        correction = [
            float(translation[0]),
            float(translation[1]),
            float(angle),
        ]
        pose = compose(correction, pose)
        error = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if abs(previous_error - error) < 1e-4:
            break
        previous_error = error

    transformed = local_to_global(source, pose)
    distances, _ = tree.query(transformed, k=1)
    valid = distances < max_dist
    if int(valid.sum()) < ICP_MIN_INLIERS:
        return pose, float('inf'), 0.0
    trimmed = np.sort(distances[valid])[:max(
        ICP_MIN_INLIERS, int(valid.sum() * ICP_TRIM)
    )]
    return (
        pose,
        float(np.sqrt(np.mean(trimmed ** 2))),
        float(valid.sum() / len(source)),
    )


def estimate_icp_information(
    target_points,
    rmse=None,
    overlap=None,
    base_weight=1.0,
    min_weight=0.05,
    max_weight=8.0,
    return_details=False,
):
    """Anisotropic SE(2) information (wx, wy, wyaw) in the target/local frame.

    Wall-like clouds (one dominant PCA axis) get low weight along the wall
    and high weight across it; yaw weight follows normal consistency.
    Quality (overlap / RMSE) scales the overall magnitude.
    """
    pts = matching_points(np.asarray(target_points, float))
    base = float(base_weight)
    details = {
        'anisotropy': 1.0,
        'along': base,
        'across': base,
        'lam_min': None,
        'lam_max': None,
        'tangent': None,
        'quality': 1.0,
    }
    if len(pts) < 20:
        w = max(min_weight, min(max_weight, base))
        info = (w, w, w)
        return (info, details) if return_details else info

    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / max(1, len(pts) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    lam_min = float(max(evals[order[0]], 1e-9))
    lam_max = float(max(evals[order[1]], 1e-9))
    # Unit vector along the dominant structure (wall tangent).
    tangent = evecs[:, order[1]]
    tx, ty = float(tangent[0]), float(tangent[1])
    # Anisotropy: ~1 isotropic, >>1 line/wall.
    anisotropy = lam_max / lam_min
    # Along-wall soft, across-wall stiff.
    along = base / math.sqrt(max(1.0, anisotropy))
    across = base * math.sqrt(max(1.0, min(anisotropy, 64.0)))
    # Map (along, across) into cartesian (wx, wy) via tangent/normal.
    # Weight matrix W = along * t t^T + across * n n^T → diagonal approx.
    nx, ny = -ty, tx
    # Effective axis-aligned info from quadratic form diagonals.
    wx = along * tx * tx + across * nx * nx
    wy = along * ty * ty + across * ny * ny
    # Yaw: walls constrain heading well when long and anisotropic.
    wyaw = base * min(4.0, 0.5 + 0.15 * math.sqrt(max(1.0, anisotropy)))

    quality = 1.0
    if overlap is not None and np.isfinite(overlap):
        quality *= float(np.clip(overlap, 0.15, 1.0))
    if rmse is not None and np.isfinite(rmse) and rmse > 1e-6:
        quality *= float(np.clip(0.08 / max(rmse, 1e-3), 0.25, 2.5))

    def _clamp(w):
        return float(np.clip(w * quality, min_weight, max_weight))

    info = (_clamp(wx), _clamp(wy), _clamp(wyaw))
    details.update({
        'anisotropy': float(anisotropy),
        'along': float(along),
        'across': float(across),
        'lam_min': lam_min,
        'lam_max': lam_max,
        'tangent': (tx, ty),
        'quality': float(quality),
        'info': info,
        'across_over_along': float(across / max(along, 1e-9)),
    })
    return (info, details) if return_details else info


def icp_to_tree(
    source, tree, initial_pose=None, max_dist=ICP_MAX_DIST, cell_m=None,
    max_source=None,
):
    """ICP against a pre-built KDTree of map points. Tree data is tree.data."""
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    max_source = 400 if max_source is None else max_source
    source = voxel_downsample(
        matching_points(np.asarray(source, float)),
        cell=cell_m,
        max_points=max_source,
    )
    if len(source) < ICP_MIN_INLIERS:
        return list(initial_pose or (0.0, 0.0, 0.0)), float('inf'), 0.0
    target = np.asarray(tree.data, float)
    pose = list(initial_pose or (0.0, 0.0, 0.0))
    previous_error = float('inf')
    for _ in range(ICP_ITERS):
        transformed = local_to_global(source, pose)
        distances, indices = tree.query(transformed, k=1)
        valid = distances < max_dist
        if int(valid.sum()) < ICP_MIN_INLIERS:
            break
        valid_indices = np.flatnonzero(valid)
        keep_count = max(ICP_MIN_INLIERS, int(len(valid_indices) * ICP_TRIM))
        keep = valid_indices[np.argsort(distances[valid_indices])[:keep_count]]
        rotation, translation, angle = rigid_fit(
            transformed[keep], target[indices[keep]]
        )
        pose = compose(
            [float(translation[0]), float(translation[1]), float(angle)],
            pose,
        )
        error = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if abs(previous_error - error) < 1e-4:
            break
        previous_error = error
    transformed = local_to_global(source, pose)
    distances, _ = tree.query(transformed, k=1)
    valid = distances < max_dist
    if int(valid.sum()) < ICP_MIN_INLIERS:
        return pose, float('inf'), 0.0
    trimmed = np.sort(distances[valid])[:max(
        ICP_MIN_INLIERS, int(valid.sum() * ICP_TRIM)
    )]
    return (
        pose,
        float(np.sqrt(np.mean(trimmed ** 2))),
        float(valid.sum() / max(1, len(source))),
    )


def motion_from_scans(
    previous_local, current_local, initial_pose=None, return_quality=False
):
    """Estimate robot motion, optionally initialized by synchronized wheels."""
    delta, error, overlap = robust_icp(
        current_local, previous_local, initial_pose=initial_pose
    )
    distance = math.hypot(delta[0], delta[1])
    angle = abs(delta[2])
    if not np.isfinite(error) or overlap < 0.35:
        result = [0.0, 0.0, 0.0]
        return (result, error, overlap) if return_quality else result
    if distance < STATIC_XY and angle < STATIC_TH:
        result = [0.0, 0.0, 0.0]
        return (result, error, overlap) if return_quality else result
    if distance > MAX_FRAME_XY:
        scale = MAX_FRAME_XY / distance
        delta[0] *= scale
        delta[1] *= scale
    delta[2] = math.copysign(min(angle, MAX_FRAME_TH), delta[2])
    return (delta, error, overlap) if return_quality else delta


def pose_changed(first, second, min_xy=KEYFRAME_XY, min_angle=KEYFRAME_TH):
    relative = relative_pose(first, second)
    return (
        math.hypot(relative[0], relative[1]) >= min_xy
        or abs(relative[2]) >= min_angle
    )


def _fit_line_pca(cloud):
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, int(np.argmax(evals))]
    projections = centered @ direction
    length = float(projections.max() - projections.min())
    angle = math.atan2(direction[1], direction[0])
    return angle, length, centroid


def _ransac_line_mask(pts, rng, dist_thresh=0.10, min_span=0.8, iters=80):
    n = len(pts)
    if n < 8:
        return None
    best_count = 0
    best_inliers = None
    for _ in range(iters):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        delta = pts[j] - pts[i]
        span = float(math.hypot(delta[0], delta[1]))
        if span < min_span:
            continue
        normal = np.array([-delta[1], delta[0]], dtype=np.float64) / span
        dist = np.abs((pts - pts[i]) @ normal)
        inliers = dist < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    return best_inliers if best_inliers is not None else None


def extract_long_lines(
    points,
    min_length=6.0,
    min_inliers=50,
    max_lines=3,
    dist_thresh=0.10,
    iters=90,
):
    """RANSAC + PCA lines in a local scan.

    Each item is (angle_rad, length_m, n_inliers, centroid_xy).
    """
    pts = matching_points(np.asarray(points, float))
    if len(pts) < min_inliers:
        return []
    rng = np.random.default_rng(0)
    remaining = np.ones(len(pts), dtype=bool)
    lines = []
    for line_index in range(max_lines):
        cloud = pts[remaining]
        if len(cloud) < min_inliers:
            break
        mask = _ransac_line_mask(
            cloud,
            rng,
            dist_thresh=dist_thresh,
            iters=iters if line_index == 0 else max(40, iters // 2),
        )
        if mask is None or int(mask.sum()) < min_inliers:
            break
        fitted = cloud[mask]
        angle, length, centroid = _fit_line_pca(fitted)
        if length < min_length:
            break
        lines.append((angle, length, int(mask.sum()), centroid))
        # Drop these inliers from the full remaining mask.
        remaining_idx = np.flatnonzero(remaining)
        remaining[remaining_idx[mask]] = False
    return lines


def longest_line_in_scan(points, dist_thresh=0.10, min_inliers=50, iters=90):
    """RANSAC line in a local scan. Returns (line_angle, length_m, n_inliers) or None."""
    lines = extract_long_lines(
        points,
        min_length=5.0,
        min_inliers=min_inliers,
        max_lines=1,
        dist_thresh=dist_thresh,
        iters=iters,
    )
    if not lines:
        return None
    angle, length, count, _centroid = lines[0]
    return angle, length, count


def manhattan_yaw_correction(local_line_angle, pose_yaw, max_corr=None):
    """Yaw delta that makes a long wall axis-aligned in the world frame."""
    max_corr = math.radians(8.0) if max_corr is None else max_corr
    world = normalize_angle(pose_yaw + local_line_angle)
    half_pi = math.pi / 2.0
    snapped = round(world / half_pi) * half_pi
    delta = normalize_angle(snapped - world)
    if abs(delta) > max_corr:
        return 0.0
    return delta


def _line_axis_and_intercept(angle, cloud, max_off_axis=None):
    max_off_axis = math.radians(8.0) if max_off_axis is None else max_off_axis
    world = normalize_angle(angle)
    half_pi = math.pi / 2.0
    if abs(abs(world) - half_pi) < max_off_axis:
        intercept = float(np.median(cloud[:, 0]))
        t0, t1 = float(cloud[:, 1].min()), float(cloud[:, 1].max())
        return 'x', intercept, t0, t1
    if min(abs(world), abs(abs(world) - math.pi)) < max_off_axis:
        intercept = float(np.median(cloud[:, 1]))
        t0, t1 = float(cloud[:, 0].min()), float(cloud[:, 0].max())
        return 'y', intercept, t0, t1
    return None


def _merge_axis_lines(lines, gap=0.16):
    merged = []
    for axis in ('x', 'y'):
        subset = [item for item in lines if item[0] == axis]
        subset.sort(key=lambda item: item[1])
        current = None
        for axis_name, intercept, t0, t1, length in subset:
            if current is None or abs(intercept - current[1]) > gap:
                if current is not None:
                    merged.append(current)
                current = [axis_name, intercept, t0, t1, length]
            else:
                n_old = max(current[4], 1.0)
                n_new = max(length, 1.0)
                current[1] = (current[1] * n_old + intercept * n_new) / (n_old + n_new)
                current[2] = min(current[2], t0)
                current[3] = max(current[3], t1)
                current[4] = n_old + n_new
        if current is not None:
            merged.append(current)
    return merged


def snap_map_to_manhattan(
    points,
    max_dist=0.18,
    min_length=3.2,
    min_inliers=35,
    max_lines=40,
    cell_m=None,
):
    """Project occupancy cells onto long axis-aligned walls.

    Does not move robot poses. Collapses thick/ghosted walls onto the
    consensus intercept so objects are drawn as straight warehouse geometry.
    """
    pts = np.asarray(points, float)
    if len(pts) < 200:
        return points, {'lines': 0, 'snapped': 0}
    cell_m = MAP_CELL if cell_m is None else float(cell_m)
    rng = np.random.default_rng(0)
    remaining = np.ones(len(pts), dtype=bool)
    raw_lines = []
    for _ in range(max_lines):
        cloud = pts[remaining]
        if len(cloud) < min_inliers:
            break
        mask = _ransac_line_mask(
            cloud, rng, dist_thresh=0.11, min_span=1.0, iters=70,
        )
        remaining_idx = np.flatnonzero(remaining)
        if mask is None or int(mask.sum()) < min_inliers:
            break
        fitted = cloud[mask]
        angle, length, _centroid = _fit_line_pca(fitted)
        remaining[remaining_idx[mask]] = False
        if length < min_length:
            continue
        parsed = _line_axis_and_intercept(angle, fitted)
        if parsed is None:
            continue
        axis, intercept, t0, t1 = parsed
        raw_lines.append([axis, intercept, t0, t1, length])
    lines = _merge_axis_lines(raw_lines, gap=0.22)
    if not lines:
        return points, {'lines': 0, 'snapped': 0}

    snapped = pts.copy()
    n_snapped = 0
    pad = 0.35
    for index, point in enumerate(pts):
        best_d = max_dist
        best = None
        for axis, intercept, t0, t1, _length in lines:
            if axis == 'x':
                if point[1] < t0 - pad or point[1] > t1 + pad:
                    continue
                dist = abs(point[0] - intercept)
                proj = (intercept, float(np.clip(point[1], t0, t1)))
            else:
                if point[0] < t0 - pad or point[0] > t1 + pad:
                    continue
                dist = abs(point[1] - intercept)
                proj = (float(np.clip(point[0], t0, t1)), intercept)
            if dist < best_d:
                best_d = dist
                best = proj
        if best is not None:
            snapped[index] = best
            n_snapped += 1
    snapped = voxel_downsample(snapped, cell=cell_m, max_points=None)
    print(
        f'  manhattan map snap: {len(lines)} walls, '
        f'{n_snapped}/{len(pts)} cells projected'
    )
    return np.asarray(snapped, float).tolist(), {
        'lines': len(lines),
        'snapped': int(n_snapped),
    }
