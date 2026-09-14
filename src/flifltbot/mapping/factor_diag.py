"""Factor-graph diagnostics vs warehouse GT.

/odom is GT only — never fed into the SLAM pipeline (USE_ABSOLUTE_ODOM stays False).
Default ``factor`` METHOD_PRESET is unchanged; ablations pass explicit kwargs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .config import METHOD_PRESETS, SUBMAP_KEYFRAMES, UP_LIDAR_Z, USE_ABSOLUTE_ODOM
from .geometry_icp import (
    estimate_icp_information,
    motion_from_scans,
    normalize_angle,
    relative_pose,
    robust_icp,
)
from .pipeline import build_global_map_from_run
from .source_io import collect_run_encoder_samples
from .trajectory_graph import make_submap

# ---- Ablation presets (all base on factor; /odom never in SLAM) ----
ABLATIONS = {
    'wheel_only': dict(
        method='factor',
        skip_lidar_sequential=True,
        skip_loop_closures=True,
        frontend_icp='wheel',
        estimate_scale=False,
        calibrate_dt=False,
    ),
    'wheel_seq': dict(
        method='factor',
        skip_lidar_sequential=False,
        skip_loop_closures=True,
        frontend_icp='scan',
        lidar_seq_target='scan',
        estimate_scale=False,
        calibrate_dt=False,
    ),
    'wheel_loops': dict(
        method='factor',
        skip_lidar_sequential=True,
        skip_loop_closures=False,
        frontend_icp='wheel',
        estimate_scale=False,
        calibrate_dt=False,
    ),
    'factor': dict(
        method='factor',
        skip_lidar_sequential=False,
        skip_loop_closures=False,
        frontend_icp='submap',
        lidar_seq_target='submap',
        submap_keyframes=10,
        estimate_scale=False,
        calibrate_dt=False,
    ),
}


def load_odom_gt(run_dir):
    """Absolute Gazebo /odom poses from encoder.jsonl — evaluation GT only."""
    samples = collect_run_encoder_samples([], Path(run_dir))
    poses = []
    for s in samples:
        if s.get('enc_available') is False:
            continue
        if not all(k in s for k in ('odom_x', 'odom_y', 'odom_yaw')):
            continue
        poses.append({
            't': float(s['enc_t']),
            'pose': [
                float(s['odom_x']),
                float(s['odom_y']),
                float(s['odom_yaw']),
            ],
            'enc_v': float(s.get('enc_v', 0.0)),
            'enc_wz': float(s.get('enc_wz', 0.0)) if s.get('enc_wz') is not None else None,
        })
    return poses


def umeyama_se2(src, dst):
    """Least-squares SE(2): dst ≈ R(src) + t. Returns (yaw, tx, ty)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    assert len(src) == len(dst) and len(src) >= 2
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s = src - mu_s
    d = dst - mu_d
    h = s.T @ d
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, sn = math.cos(yaw), math.sin(yaw)
    tx = mu_d[0] - (c * mu_s[0] - sn * mu_s[1])
    ty = mu_d[1] - (sn * mu_s[0] + c * mu_s[1])
    return yaw, tx, ty


def apply_se2(pts, yaw, tx, ty):
    pts = np.asarray(pts, float)
    c, s = math.cos(yaw), math.sin(yaw)
    x = tx + c * pts[:, 0] - s * pts[:, 1]
    y = ty + s * pts[:, 0] + c * pts[:, 1]
    return np.column_stack([x, y])


# Gazebo enc_wz on this cartbot tops out ~0.15 rad/s; use 0.08 so turns are labeled.
TURN_WZ_THRESH = 0.08


def _is_turn_sample(sample, prev_sample=None):
    wz = sample.get('enc_wz')
    if wz is not None and abs(float(wz)) > TURN_WZ_THRESH:
        return True
    if prev_sample is not None:
        dt = float(sample['t']) - float(prev_sample['t'])
        if 1e-3 < dt < 0.5:
            dyaw = normalize_angle(
                float(sample['pose'][2]) - float(prev_sample['pose'][2])
            )
            if abs(dyaw / dt) > TURN_WZ_THRESH:
                return True
    return False


def sync_est_to_gt(est_poses, gt_odom, max_dt=0.25):
    """Match estimated poses (with 't') to nearest GT odom by time."""
    empty = (
        np.empty((0, 2)), np.empty((0, 2)),
        np.empty(0), np.empty(0), [],
    )
    if not est_poses or not gt_odom:
        return empty
    gt_t = np.asarray([g['t'] for g in gt_odom], float)
    gt_xy = np.asarray([g['pose'][:2] for g in gt_odom], float)
    gt_yaw = np.asarray([g['pose'][2] for g in gt_odom], float)
    timed = [e for e in est_poses if e.get('t') is not None]
    est_xy, dst_xy, est_yaw, dst_yaw, labels = [], [], [], [], []
    if timed:
        for ep in timed:
            t = ep['t']
            j = int(np.argmin(np.abs(gt_t - t)))
            if abs(gt_t[j] - t) > max_dt:
                continue
            est_xy.append(ep['pose'][:2])
            dst_xy.append(gt_xy[j])
            est_yaw.append(ep['pose'][2])
            dst_yaw.append(gt_yaw[j])
            prev = gt_odom[j - 1] if j > 0 else None
            labels.append(
                'turn' if _is_turn_sample(gt_odom[j], prev) else 'straight'
            )
    else:
        # Index alignment fallback (equal-length trajectories).
        n = min(len(est_poses), len(gt_odom))
        for i in range(n):
            est_xy.append(est_poses[i]['pose'][:2])
            dst_xy.append(gt_xy[i])
            est_yaw.append(est_poses[i]['pose'][2])
            dst_yaw.append(gt_yaw[i])
            prev = gt_odom[i - 1] if i > 0 else None
            labels.append(
                'turn' if _is_turn_sample(gt_odom[i], prev) else 'straight'
            )
    if not est_xy:
        return empty
    return (
        np.asarray(est_xy, float),
        np.asarray(dst_xy, float),
        np.asarray(est_yaw, float),
        np.asarray(dst_yaw, float),
        labels,
    )


def ate_metrics(est_xy, gt_xy, est_yaw=None, gt_yaw=None, labels=None):
    if len(est_xy) < 2:
        return {'n': 0}
    yaw, tx, ty = umeyama_se2(est_xy, gt_xy)
    aligned = apply_se2(est_xy, yaw, tx, ty)
    err = np.linalg.norm(aligned - gt_xy, axis=1)
    out = {
        'n': int(len(err)),
        'ate_rmse_m': float(np.sqrt(np.mean(err ** 2))),
        'ate_median_m': float(np.median(err)),
        'ate_p95_m': float(np.percentile(err, 95)),
        'ate_max_m': float(np.max(err)),
        'align_yaw_deg': float(math.degrees(yaw)),
        'align_tx_m': float(tx),
        'align_ty_m': float(ty),
    }
    if est_yaw is not None and gt_yaw is not None and len(est_yaw) == len(gt_yaw):
        # Rotate estimated yaw by Umeyama yaw.
        yaw_err = [
            abs(normalize_angle(ey + yaw - gy))
            for ey, gy in zip(est_yaw, gt_yaw)
        ]
        out['ate_yaw_rmse_deg'] = float(
            math.degrees(math.sqrt(np.mean(np.square(yaw_err))))
        )
        out['ate_yaw_median_deg'] = float(math.degrees(np.median(yaw_err)))
    if labels is not None and len(labels) == len(err):
        for kind in ('straight', 'turn'):
            mask = np.asarray([l == kind for l in labels])
            if mask.any():
                e = err[mask]
                out[f'ate_rmse_{kind}_m'] = float(np.sqrt(np.mean(e ** 2)))
                out[f'ate_median_{kind}_m'] = float(np.median(e))
                out[f'n_{kind}'] = int(mask.sum())
    return out


def path_length(xy):
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def rpe_at_distances(est_xy, gt_xy, est_yaw, gt_yaw, distances=(1.0, 5.0, 10.0)):
    """Relative pose error at path distances (Umeyama-aligned first)."""
    if len(est_xy) < 3:
        return {}
    yaw, tx, ty = umeyama_se2(est_xy, gt_xy)
    a_xy = apply_se2(est_xy, yaw, tx, ty)
    a_yaw = est_yaw + yaw
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a_xy, axis=0), axis=1))])
    out = {}
    for d in distances:
        trans, rots = [], []
        j = 0
        for i in range(len(a_xy)):
            while j < len(a_xy) - 1 and cum[j] - cum[i] < d:
                j += 1
            if j >= len(a_xy) or cum[j] - cum[i] < d * 0.9:
                continue
            # Relative poses over segment i→j
            e_rel = relative_pose(
                [a_xy[i, 0], a_xy[i, 1], a_yaw[i]],
                [a_xy[j, 0], a_xy[j, 1], a_yaw[j]],
            )
            g_rel = relative_pose(
                [gt_xy[i, 0], gt_xy[i, 1], gt_yaw[i]],
                [gt_xy[j, 0], gt_xy[j, 1], gt_yaw[j]],
            )
            err = relative_pose(e_rel, g_rel)
            # err = inv(e_rel) ∘ g_rel would be better; use pose difference magnitude
            dx = e_rel[0] - g_rel[0]
            dy = e_rel[1] - g_rel[1]
            dth = normalize_angle(e_rel[2] - g_rel[2])
            trans.append(math.hypot(dx, dy))
            rots.append(abs(dth))
        if trans:
            out[f'rpe_{d:g}m_trans_rmse'] = float(np.sqrt(np.mean(np.square(trans))))
            out[f'rpe_{d:g}m_trans_median'] = float(np.median(trans))
            out[f'rpe_{d:g}m_rot_rmse_deg'] = float(
                math.degrees(math.sqrt(np.mean(np.square(rots))))
            )
            out[f'rpe_{d:g}m_n'] = int(len(trans))
    return out


def poses_from_build(result):
    """Extract timed poses from build_global_map_from_run result."""
    pose_records = result.get('poses') or []
    out = []
    for rec in pose_records:
        if not isinstance(rec, dict):
            continue
        pose = rec.get('pose')
        if pose is None and all(k in rec for k in ('x', 'y', 'yaw')):
            pose = [rec['x'], rec['y'], rec['yaw']]
        t = rec.get('t')
        if t is None:
            t = rec.get('frame_t')
        if pose is None:
            continue
        out.append({'t': t, 'pose': list(pose)})
    # Fallback: map JSON
    if not out and result.get('out_json'):
        path = Path(result['out_json'])
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            for rec in data.get('poses') or []:
                if isinstance(rec, dict) and 'pose' in rec:
                    out.append({
                        't': rec.get('t') or rec.get('frame_t'),
                        'pose': list(rec['pose']),
                    })
                elif isinstance(rec, (list, tuple)) and len(rec) >= 3:
                    out.append({'t': None, 'pose': list(rec[:3])})
    return out


def warehouse_gt_meta(lidar_z=None):
    """Warehouse GT visibility note for horizontal lidar at height z."""
    z = UP_LIDAR_Z if lidar_z is None else float(lidar_z)
    return {
        'gt': 'warehouse walls + shelf footprints (accuracy_eval.gt_segments)',
        'lidar_z_m': z,
        'visibility': (
            'Warehouse shelves/walls span floor→ceiling in Gazebo; '
            f'horizontal beam at z≈{z:.2f} m (up/horizon) sees the same 2D GT '
            'footprint as score_map. No short furniture to filter.'
        ),
        'score_fn': 'runtime.accuracy_eval.score_map (map cells → world via SPAWN)',
        'odom_gt': (
            'Gazebo /odom (odom_x,y,yaw) from encoder.jsonl — post-build only; '
            f'USE_ABSOLUTE_ODOM={USE_ABSOLUTE_ODOM}'
        ),
    }


def evaluate_build(result, run_dir, gt_odom=None):
    from runtime.accuracy_eval import score_map

    gt_odom = gt_odom if gt_odom is not None else load_odom_gt(run_dir)
    est = poses_from_build(result)
    est_xy, gt_xy, est_yaw, gt_yaw, labels = sync_est_to_gt(est, gt_odom)
    ate = ate_metrics(est_xy, gt_xy, est_yaw, gt_yaw, labels)
    rpe = rpe_at_distances(est_xy, gt_xy, est_yaw, gt_yaw) if len(est_xy) >= 3 else {}
    map_score = score_map(result['points'])
    stats = result.get('stats') or {}
    n_frames = int(stats.get('n_frames') or 0)
    wall_s = result.get('wall_s')
    if wall_s is None and isinstance(stats.get('timing_s'), dict):
        wall_s = stats['timing_s'].get('total')
    wall_s = float(wall_s) if wall_s is not None else None
    runtime = {
        'wall_s': wall_s,
        'n_frames': n_frames,
        'ms_per_frame': (
            float(wall_s) * 1000.0 / n_frames if wall_s and n_frames else None
        ),
        'fps': (n_frames / float(wall_s) if wall_s and n_frames else None),
        'timing_s': stats.get('timing_s'),
    }
    return {
        'stats': {
            k: stats.get(k)
            for k in (
                'mode', 'factor_mode', 's_v', 'estimate_scale',
                'lidar_odom_dt_s', 'wheel_edges', 'lidar_seq_edges',
                'loop_closures', 'pose_graph_cost', 'stable_cells',
                'loop_error_m', 'timing_s', 'n_frames', 'keyframes',
                'lidar_seq_rmse_median', 'lidar_seq_overlap_median',
                'lidar_seq_across_over_along_median', 'loop_rmse_median',
                'lidar_seq_target', 'submap_keyframes',
                'frontend_hardening', 'seq_max_rmse', 'loop_best_of_k',
            )
        },
        'ate': ate,
        'rpe': rpe,
        'map_error': map_score,
        'runtime': runtime,
        'gt_meta': warehouse_gt_meta(),
        'odom_path_m': path_length(gt_xy) if len(gt_xy) else 0.0,
        'wall_s': wall_s,
        'map_png': str(result.get('out_png', '')),
        'map_json': str(result.get('out_json', '')),
    }


def format_ablation_table(variants):
    """Compact text table for ablation / submap rows."""
    lines = [
        f"{'tag':16s} {'ATE':>8s} {'med':>7s} {'RPE1':>6s} {'RPE5':>6s} "
        f"{'RPE10':>6s} {'mapcm':>6s} {'seq':>5s} {'loops':>5s} "
        f"{'ms/fr':>7s} {'FPS':>6s} {'s_v':>6s}"
    ]
    for v in variants:
        a = v.get('ate') or {}
        r = v.get('rpe') or {}
        s = v.get('stats') or {}
        m = v.get('map_error') or {}
        rt = v.get('runtime') or {}
        lines.append(
            f"{str(v.get('tag', '?')):16s} "
            f"{(a.get('ate_rmse_m') or float('nan')):8.4f} "
            f"{(a.get('ate_median_m') or float('nan')):7.4f} "
            f"{(r.get('rpe_1m_trans_rmse') or float('nan')):6.3f} "
            f"{(r.get('rpe_5m_trans_rmse') or float('nan')):6.3f} "
            f"{(r.get('rpe_10m_trans_rmse') or float('nan')):6.3f} "
            f"{(m.get('near_gt_median_cm') or float('nan')):6.1f} "
            f"{int(s.get('lidar_seq_edges') or 0):5d} "
            f"{int(s.get('loop_closures') or 0):5d} "
            f"{(rt.get('ms_per_frame') or float('nan')):7.1f} "
            f"{(rt.get('fps') or float('nan')):6.2f} "
            f"{(s.get('s_v') if s.get('s_v') is not None else float('nan')):6.4f}"
        )
    return '\n'.join(lines)


def run_ablation(
    run_dir, cameras=None, source='sim', tags=None, out_dir=None, max_frames=None,
):
    """Build 4 ablations + score vs warehouse GT and /odom ATE."""
    run_dir = Path(run_dir)
    cameras = cameras or ['horizon']
    out_dir = Path(out_dir) if out_dir else (
        Path('results') / run_dir.parts[-3] / run_dir.parts[-2] / run_dir.name
        if len(run_dir.parts) >= 3
        else run_dir.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    assert USE_ABSOLUTE_ODOM is False, '/odom must not enter SLAM'
    gt_odom = load_odom_gt(run_dir)
    tags = tags or list(ABLATIONS.keys())
    rows = []
    for tag in tags:
        cfg = dict(ABLATIONS[tag])
        method = cfg.pop('method')
        if max_frames is not None:
            cfg['max_frames'] = int(max_frames)
        print('=' * 64)
        print(f'ABLATION {tag}  kwargs={cfg}')
        t0 = time.perf_counter()
        result = build_global_map_from_run(
            run_dir,
            method=method,
            cameras=cameras,
            tag=f'{run_dir.name}_{tag}',
            source=source,
            **cfg,
        )
        result['wall_s'] = time.perf_counter() - t0
        row = {'tag': tag, 'kwargs': cfg, **evaluate_build(result, run_dir, gt_odom)}
        rows.append(row)
        print(json.dumps({
            'tag': tag,
            'ate_rmse_m': row['ate'].get('ate_rmse_m'),
            'map_near_gt_median_cm': row['map_error'].get('near_gt_median_cm'),
            'lidar_seq': row['stats'].get('lidar_seq_edges'),
            'loops': row['stats'].get('loop_closures'),
            's_v': row['stats'].get('s_v'),
            'ms_per_frame': (row.get('runtime') or {}).get('ms_per_frame'),
            'fps': (row.get('runtime') or {}).get('fps'),
        }, indent=2))
    print(format_ablation_table(rows))
    summary = {
        'run_dir': str(run_dir),
        'gt_odom_n': len(gt_odom),
        'use_absolute_odom_in_slam': USE_ABSOLUTE_ODOM,
        'max_frames': max_frames,
        'variants': rows,
        'table': format_ablation_table(rows),
        'question': (
            'Why does sequential LiDAR barely help (factor ≈ factor_nolidar_seq)? '
            'Does scan-to-submap improve over scan-to-scan?'
        ),
    }
    path = out_dir / 'factor_ablation.json'
    path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {path}')
    return summary


def probe_icp_anisotropy(run_dir, cameras=None, source='sim', max_pairs=80):
    """Sample sequential ICP quality + JᵀJ / anisotropy along walls."""
    from .pipeline import sew_all_local_frames

    cameras = cameras or ['horizon']
    # Load a short wheel-only pass to get deskewed frames cheaply.
    _pts, _poses, _files, _stats, frames, _p = sew_all_local_frames(
        run_dir,
        **METHOD_PRESETS['factor'],
        cameras=cameras,
        source=source,
        return_internals=True,
        skip_lidar_sequential=True,
        skip_loop_closures=True,
        estimate_scale=False,
        calibrate_dt=False,
        frontend_icp='wheel',
        max_frames=400,
    )
    rows = []
    step = max(1, len(frames) // max_pairs)
    for i in range(0, len(frames) - 1, step):
        a, b = frames[i], frames[i + 1]
        if len(a) < 80 or len(b) < 80:
            continue
        _meas, rmse, overlap = motion_from_scans(a, b, return_quality=True)
        info, details = estimate_icp_information(
            a, rmse=rmse, overlap=overlap, return_details=True,
        )
        rows.append({
            'i': i,
            'rmse': float(rmse) if np.isfinite(rmse) else None,
            'overlap': float(overlap),
            'info_wx_wy_wyaw': list(info),
            'anisotropy': details.get('anisotropy'),
            'across_over_along': details.get('across_over_along'),
            'tangent': details.get('tangent'),
        })
    anis = [r['across_over_along'] for r in rows if r.get('across_over_along')]
    return {
        'n_pairs': len(rows),
        'median_across_over_along': float(np.median(anis)) if anis else None,
        'frac_anisotropic_gt_2': (
            float(np.mean([a > 2 for a in anis])) if anis else None
        ),
        'note': (
            'across_over_along >> 1 means wall-like: uncertainty along wall '
            'higher (lower info) than across — anisotropic observability.'
        ),
        'samples': rows[:40],
    }


def compare_scan_vs_submap(
    run_dir, cameras=None, source='sim', submap_ns=None, max_frames=None,
    with_loops=True,
):
    """Ablation: sequential ICP target = scan vs submap (N keyframes).

    with_loops=True matches production factor (seq + LiDAR loops).
    """
    cameras = cameras or ['horizon']
    submap_ns = submap_ns or [10, 18, 25, 40]
    gt_odom = load_odom_gt(run_dir)
    rows = []
    for target, n in [('scan', None)] + [('submap', n) for n in submap_ns]:
        tag = f"{'factor' if with_loops else 'seq'}_{target}" + (
            f'_n{n}' if n else ''
        )
        kwargs = dict(
            skip_loop_closures=not with_loops,
            skip_lidar_sequential=False,
            frontend_icp='scan' if target == 'scan' else 'submap',
            lidar_seq_target=target,
            estimate_scale=False,
            calibrate_dt=False,
        )
        if n is not None:
            kwargs['submap_keyframes'] = int(n)
        if max_frames is not None:
            kwargs['max_frames'] = int(max_frames)
        print(f'ICP target experiment: {tag}')
        t0 = time.perf_counter()
        result = build_global_map_from_run(
            run_dir,
            method='factor',
            cameras=cameras,
            tag=f'{Path(run_dir).name}_{tag}',
            source=source,
            **kwargs,
        )
        result['wall_s'] = time.perf_counter() - t0
        row = {
            'tag': tag,
            'lidar_seq_target': target,
            'submap_keyframes': n,
            **evaluate_build(result, run_dir, gt_odom),
        }
        print(json.dumps({
            'tag': tag,
            'ate_rmse_m': row['ate'].get('ate_rmse_m'),
            'lidar_seq_rmse_median': row['stats'].get('lidar_seq_rmse_median'),
            'ms_per_frame': (row.get('runtime') or {}).get('ms_per_frame'),
            'fps': (row.get('runtime') or {}).get('fps'),
            'wall_s': row['wall_s'],
        }, indent=2))
        rows.append(row)
    print(format_ablation_table(rows))
    return {'variants': rows, 'table': format_ablation_table(rows), 'with_loops': with_loops}


def sweep_temporal_offset(
    run_dir,
    cameras=None,
    source='sim',
    dt_min=-0.15,
    dt_max=0.15,
    dt_step=0.01,
    max_frames=None,
):
    """ATE vs fixed δt (encoder shift). /odom not used in SLAM."""
    cameras = cameras or ['horizon']
    gt_odom = load_odom_gt(run_dir)
    curve = []
    dt = dt_min
    while dt <= dt_max + 1e-12:
        print(f'δt sweep {dt*1000:.0f} ms')
        t0 = time.perf_counter()
        kwargs = dict(
            calibrate_dt=False,
            encoder_dt_s=dt,
            estimate_scale=False,
            skip_loop_closures=True,  # isolate frontend+seq effect
        )
        if max_frames is not None:
            kwargs['max_frames'] = int(max_frames)
        result = build_global_map_from_run(
            run_dir,
            method='factor',
            cameras=cameras,
            tag=f'{Path(run_dir).name}_dt{int(round(dt*1000))}',
            source=source,
            **kwargs,
        )
        result['wall_s'] = time.perf_counter() - t0
        ev = evaluate_build(result, run_dir, gt_odom)
        curve.append({
            'dt_s': round(dt, 5),
            'dt_ms': round(dt * 1000, 1),
            'ate_rmse_m': ev['ate'].get('ate_rmse_m'),
            'ate_median_m': ev['ate'].get('ate_median_m'),
            'map_near_gt_median_cm': ev['map_error'].get('near_gt_median_cm'),
            'lidar_seq_edges': ev['stats'].get('lidar_seq_edges'),
            'lidar_seq_rmse_median': ev['stats'].get('lidar_seq_rmse_median'),
            'wall_s': result['wall_s'],
        })
        dt += dt_step
    valid = [c for c in curve if c['ate_rmse_m'] is not None]
    best = min(valid, key=lambda c: c['ate_rmse_m']) if valid else None
    best_icp = (
        min(
            [c for c in valid if c.get('lidar_seq_rmse_median') is not None],
            key=lambda c: c['lidar_seq_rmse_median'],
        )
        if any(c.get('lidar_seq_rmse_median') is not None for c in valid)
        else None
    )
    return {
        'curve': curve,
        'best_by_ate': best,
        'best_by_icp_residual': best_icp,
        'hypothesis_ms': -70,
        'note': 'Minimum ATE should be stable near −70 ms if prior calib holds.',
    }


def sweep_scale(run_dir, cameras=None, source='sim', max_frames=None):
    """s_v=1 vs optimized s_v."""
    cameras = cameras or ['horizon']
    gt_odom = load_odom_gt(run_dir)
    rows = []
    for tag, kwargs in (
        ('s_v_fixed_1', dict(estimate_scale=False)),
        ('s_v_opt', dict(estimate_scale=True)),
    ):
        if max_frames is not None:
            kwargs['max_frames'] = int(max_frames)
        result = build_global_map_from_run(
            run_dir,
            method='factor',
            cameras=cameras,
            tag=f'{Path(run_dir).name}_{tag}',
            source=source,
            **kwargs,
        )
        rows.append({
            'tag': tag,
            **evaluate_build(result, run_dir, gt_odom),
        })
    return {'variants': rows}


def diagnose_lidar_frontend(ablation_summary):
    """If wheel_seq ≈ wheel_only, flag frontend issues."""
    by = {r['tag']: r for r in ablation_summary.get('variants', [])}
    w = by.get('wheel_only', {}).get('ate', {})
    s = by.get('wheel_seq', {}).get('ate', {})
    f = by.get('factor', {}).get('ate', {})
    wl = by.get('wheel_loops', {}).get('ate', {})
    out = {'checks': []}
    if w.get('ate_rmse_m') is not None and s.get('ate_rmse_m') is not None:
        delta = abs(s['ate_rmse_m'] - w['ate_rmse_m'])
        out['wheel_seq_minus_wheel_only_ate_rmse'] = delta
        if delta < 0.05:
            out['checks'].append(
                'sequential ICP barely changes ATE vs wheel_only — inspect '
                'deskew, δt sync, extrinsics, ICP correspondences/degeneracy'
            )
    if f.get('ate_rmse_m') is not None and wl.get('ate_rmse_m') is not None:
        out['factor_minus_wheel_loops_ate_rmse'] = (
            f['ate_rmse_m'] - wl['ate_rmse_m']
        )
    out['stats_compare'] = {
        tag: by[tag]['stats']
        for tag in ('wheel_only', 'wheel_seq', 'wheel_loops', 'factor')
        if tag in by
    }
    return out


def run_full_diagnostics(
    run_dir,
    cameras=None,
    source='sim',
    out_dir=None,
    do_dt_sweep=True,
    do_submap=True,
    do_scale=True,
    do_anisotropy=True,
    max_frames=None,
):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else (
        Path('results') / run_dir.parts[-3] / run_dir.parts[-2] / run_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'run_dir': str(run_dir),
        'use_absolute_odom_in_slam': USE_ABSOLUTE_ODOM,
        'max_frames': max_frames,
        'gt_meta': warehouse_gt_meta(),
    }
    report['ablations'] = run_ablation(
        run_dir, cameras=cameras, source=source, out_dir=out_dir,
        max_frames=max_frames,
    )
    report['frontend_diagnosis'] = diagnose_lidar_frontend(report['ablations'])
    if do_anisotropy:
        report['icp_anisotropy'] = probe_icp_anisotropy(
            run_dir, cameras=cameras, source=source,
        )
    if do_submap:
        report['scan_vs_submap'] = compare_scan_vs_submap(
            run_dir, cameras=cameras, source=source, max_frames=max_frames,
        )
        if out_dir:
            (out_dir / 'factor_submap.json').write_text(
                json.dumps(report['scan_vs_submap'], indent=2) + '\n',
                encoding='utf-8',
            )
    if do_scale:
        report['scale_sweep'] = sweep_scale(
            run_dir, cameras=cameras, source=source, max_frames=max_frames,
        )
        if out_dir:
            (out_dir / 'factor_scale.json').write_text(
                json.dumps(report['scale_sweep'], indent=2) + '\n',
                encoding='utf-8',
            )
    if do_dt_sweep:
        report['dt_sweep'] = sweep_temporal_offset(
            run_dir, cameras=cameras, source=source, max_frames=max_frames,
        )
        if out_dir:
            (out_dir / 'dt_sweep.json').write_text(
                json.dumps(report['dt_sweep'], indent=2) + '\n',
                encoding='utf-8',
            )
    path = out_dir / 'factor_diagnostics.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote full report {path}')
    return report
