#!/usr/bin/env python3
"""Notebook-style marker reconstruction using only move-log nav_* poses.

The runtime geometry intentionally follows ``ray_plane_intersection_plot.ipynb``:

    P_local = O + ((H - O_z) / d_z) * d
    P_global_xy = Rz(nav_dgr + heading_offset) @ P_local_xy + [nav_x, nav_y]

Only ``nav_t``, ``nav_x``, ``nav_y`` and ``nav_dgr`` are read from the move
logger.  Cartographer, GT, and any t/x/y/dgr columns are not runtime inputs.
GT is loaded only by the final offline plot/metrics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORDING = ROOT / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
DEFAULT_GT = ROOT / "ros-data/gt/real_room/real_room/ceiling_labels_room_real_gt.json"
DEFAULT_OUT = ROOT / "ros-data/results/real/20260914/nav_marker_map_notebook_style"


def parse_utc(value: str) -> float:
    value = str(value).strip()
    parsed = dt.datetime.strptime(value, "%Y_%m_%d_%H_%M_%S_%f")
    return parsed.replace(tzinfo=dt.timezone.utc).timestamp()


def load_frame_times(path: Path, count: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        result = np.asarray([float(row["time_take_img_sec"]) for row in csv.DictReader(stream)], dtype=float)
    if len(result) != count or not np.isfinite(result).all():
        raise ValueError(f"frame timestamp count mismatch: {path} ({len(result)} != {count})")
    return result


def load_camera_origin(path: Path) -> np.ndarray:
    line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    origin = np.asarray([float(v) for v in line.split()[:3]], dtype=float) / 100.0
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError(f"invalid camera position: {path}")
    return origin


def load_nav(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load exactly nav_t/nav_x/nav_y/nav_dgr; all other pose columns are forbidden."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                # Deliberately do not access t/x/y/dgr or any encoder field.
                stamp = parse_utc(row["nav_t"])
                x, y, yaw = float(row["nav_x"]), float(row["nav_y"]), float(row["nav_dgr"])
            except (KeyError, TypeError, ValueError):
                continue
            # The logger's epoch-zero sentinel is 1970-01-01 03:00.
            if stamp < 1.0e9 or not all(math.isfinite(v) for v in (stamp, x, y, yaw)):
                continue
            rows.append((stamp, x, y, math.radians(yaw)))
    if len(rows) < 2:
        raise ValueError(f"not enough valid nav rows in {path}")
    a = np.asarray(sorted(rows), dtype=float)
    # Remove duplicate timestamps while preserving the last sample.
    _, keep = np.unique(a[:, 0][::-1], return_index=True)
    a = a[len(a) - 1 - keep]
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:], {"source": str(path.resolve()), "fields": ["nav_t", "nav_x", "nav_y", "nav_dgr"], "forbidden_fields": ["t", "x", "y", "dgr"], "timestamp_timezone": "UTC", "rows": int(len(a))}


def interp_nav(query: float, times: np.ndarray, poses: np.ndarray) -> np.ndarray | None:
    i = int(np.searchsorted(times, query))
    if i == 0 or i >= len(times):
        return None
    dtv = times[i] - times[i - 1]
    w = (query - times[i - 1]) / dtv if dtv > 0 else 0.0
    delta = (poses[i, 2] - poses[i - 1, 2] + math.pi) % (2 * math.pi) - math.pi
    return np.asarray([poses[i - 1, 0] + w * (poses[i, 0] - poses[i - 1, 0]), poses[i - 1, 1] + w * (poses[i, 1] - poses[i - 1, 1]), poses[i - 1, 2] + w * delta])


def rot(theta: float) -> np.ndarray:
    return np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=float)


def raw_rows(payload: dict[str, Any], frame_times: np.ndarray, nav_t: np.ndarray, nav_pose: np.ndarray, origin: np.ndarray, height: float, heading_offset_deg: float) -> list[dict[str, Any]]:
    result = []
    heading_offset = math.radians(heading_offset_deg)
    for frame_index, (frame, stamp) in enumerate(zip(payload.get("data", []), frame_times)):
        pose = interp_nav(float(stamp), nav_t, nav_pose)
        if pose is None or not isinstance(frame, dict):
            continue
        R = rot(float(pose[2]) + heading_offset)
        for track_id, item in frame.items():
            if not isinstance(item, dict):
                continue
            try:
                code = int(item["code"])
                ray = np.asarray(item["ray"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if code < 0 or ray.shape != (3,) or not np.isfinite(ray).all() or abs(ray[2]) < 1e-9:
                continue
            depth = (height - origin[2]) / ray[2]
            if depth <= 0 or not math.isfinite(depth):
                continue
            local = origin + depth * ray
            point = R @ local[:2] + pose[:2]
            result.append({"frame_index": frame_index, "timestamp_s": float(stamp), "track_id": str(track_id), "marker_id": code, "ray": ray.tolist(), "pose": [float(v) for v in pose], "point_xy_m": [float(v) for v in point], "depth_m": float(depth)})
    return result


def robust_code_aggregates(rows: list[dict[str, Any]], min_observations: int, gate_m: float) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    by_code: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_code[int(row["marker_id"])].append(row)
    aggregates = {}
    rejected = []
    for code, values in sorted(by_code.items()):
        points = np.asarray([r["point_xy_m"] for r in values], dtype=float)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        mad = float(np.median(np.abs(distances - np.median(distances))))
        threshold = max(gate_m, float(np.median(distances)) + 3.0 * 1.4826 * mad)
        keep = distances <= threshold
        if int(keep.sum()) < min_observations:
            rejected.append({"marker_id": code, "reason": "below_min_after_robust_filter", "raw": len(values), "kept": int(keep.sum())})
            continue
        kept = [values[i] for i in np.flatnonzero(keep)]
        center = np.median(np.asarray([r["point_xy_m"] for r in kept]), axis=0)
        aggregates[code] = {"marker_id": code, "point_xy_m": [float(v) for v in center], "raw_observations": len(values), "filtered_observations": len(kept), "filter_threshold_m": float(threshold), "spread_m": float(np.median(np.linalg.norm(np.asarray([r["point_xy_m"] for r in kept]) - center, axis=1))), "track_ids": sorted({str(r["track_id"]) for r in kept})}
    return aggregates, rejected


def cluster_aggregates(aggregates: dict[int, dict], radius_m: float) -> list[dict[str, Any]]:
    """Cluster per-code centres so decoder identity switches remain one physical point."""
    codes = sorted(aggregates)
    if not codes:
        return []
    pts = np.asarray([aggregates[c]["point_xy_m"] for c in codes], dtype=float)
    seen = set(); clusters = []
    for seed in range(len(codes)):
        if seed in seen: continue
        stack = [seed]; seen.add(seed); members = []
        while stack:
            i = stack.pop(); members.append(i)
            near = np.flatnonzero(np.linalg.norm(pts - pts[i], axis=1) <= radius_m)
            for j in near:
                j = int(j)
                if j not in seen: seen.add(j); stack.append(j)
        member_codes = [codes[i] for i in members]
        weights = np.asarray([aggregates[c]["filtered_observations"] for c in member_codes], dtype=float)
        center = np.median(pts[members], axis=0)
        clusters.append({"cluster_id": len(clusters) + 1, "candidate_marker_ids": member_codes, "point_xy_m": [float(v) for v in center], "aggregated_codes": len(member_codes), "filtered_observations": int(weights.sum()), "spread_m": float(np.max(np.linalg.norm(pts[members] - center, axis=1))) if len(members) else 0.0})
    return clusters


def triangulate_height(rows: list[dict[str, Any]], origin: np.ndarray, min_obs: int, gate_m: float, heading_offset_deg: float) -> dict[str, Any] | None:
    """Estimate z from nav-transformed 3-D rays, without a map or GT."""
    by_code: dict[int, list[dict[str, Any]]] = defaultdict(list)
    off = math.radians(heading_offset_deg)
    for r in rows: by_code[int(r["marker_id"])].append(r)
    zs = []
    for code, values in by_code.items():
        if len(values) < min_obs: continue
        origins, dirs = [], []
        for r in values[::max(1, len(values)//80)]:
            p = np.asarray(r["pose"]); R = rot(float(p[2]) + off); ray = np.asarray(r["ray"]); origins.append(np.r_[R @ origin[:2] + p[:2], origin[2]]); dirs.append(np.r_[R @ ray[:2], ray[2]])
        origins, dirs = np.asarray(origins), np.asarray(dirs); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        best = None
        for i in range(len(dirs)-1):
            for j in range(i+1, len(dirs)):
                d1, d2 = dirs[i], dirs[j]; w = origins[j] - origins[i]; den = 1 - float(d1 @ d2) ** 2
                if den < 1e-6: continue
                a = (w @ d1 - (d1 @ d2) * (w @ d2)) / den; b = ((d1 @ d2) * (w @ d1) - w @ d2) / den
                q = .5 * (origins[i] + a*d1 + origins[j] + b*d2)
                residual = np.linalg.norm(np.cross(q[None,:] - origins, dirs), axis=1); depths = np.sum((q[None,:]-origins)*dirs, axis=1); inl = (residual <= gate_m) & (depths > 0)
                score = (int(inl.sum()), -float(np.median(residual[inl])) if inl.any() else -1)
                if best is None or score > best[0]: best = (score, q, inl)
        if best is not None and best[2].sum() >= 4: zs.append(float(best[1][2]))
    if len(zs) < 3: return {"status": "abstained", "reason": "fewer_than_three_independent_height_fits", "heights_m": zs}
    z = float(np.median(zs)); mad = float(np.median(np.abs(np.asarray(zs)-z))); keep = np.abs(np.asarray(zs)-z) <= max(.12, 3*1.4826*mad)
    if int(keep.sum()) < 3: return {"status": "abstained", "reason": "height_fits_inconsistent", "heights_m": zs}
    robust_sigma = float(1.4826*np.median(np.abs(np.asarray(zs)[keep]-np.median(np.asarray(zs)[keep]))))
    # A V6-style estimate is actionable only when independent marker fits
    # agree.  This guard prevents a broad mixture of wrong ray/ID hypotheses
    # from being presented as an automatic ceiling height.
    if robust_sigma > 0.15:
        return {"status": "abstained", "reason": "height_fits_have_large_robust_sigma", "heights_m": zs, "inlier_fits": int(keep.sum()), "robust_sigma_m": robust_sigma}
    return {"status": "accepted", "ceiling_height_m": float(np.median(np.asarray(zs)[keep])), "heights_m": zs, "inlier_fits": int(keep.sum()), "robust_sigma_m": robust_sigma}


def load_gt(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8")); result = {}
    for item in payload.get("labels", []):
        # Notebook/global nav convention is the source encoder tag frame,
        # whose x/y are stored in cm and whose origin is marker 98.
        e = item.get("encoder_tag_cm", {}); result[int(item["id"])] = np.asarray([float(e["x"])/100, float(e["y"])/100])
    return result


def box_distance(point: np.ndarray, center: np.ndarray, side_m: float = .06) -> float:
    delta = np.maximum(np.abs(point-center)-side_m/2, 0.0)
    return float(np.linalg.norm(delta))


def metrics(pred: list[dict], gt: dict[int, np.ndarray]) -> dict[str, Any]:
    p = np.asarray([x["point_xy_m"] for x in pred], dtype=float); g = np.asarray(list(gt.values()), dtype=float)
    if not len(p) or not len(g): return {"prediction_count": len(p), "gt_count": len(g)}
    d = np.linalg.norm(p[:,None,:]-g[None,:,:],axis=2); rr,cc=linear_sum_assignment(d); errors=d[rr,cc]
    rows=[]
    for i,j,e in zip(rr,cc,errors): rows.append({"prediction_index":int(i),"nearest_gt_id":list(gt)[int(j)],"center_error_m":float(e),"distance_to_gt_box_m":box_distance(p[i],g[j])})
    return {"prediction_count":int(len(p)),"gt_count":int(len(g)),"center_error_m":{"mae":float(errors.mean()),"rmse":float(np.sqrt(np.mean(errors**2))),"median":float(np.median(errors)),"p95":float(np.percentile(errors,95)),"max":float(errors.max())},"rows":rows}


def render(path: Path, raw: list[dict], pred: list[dict], gt: dict[int, np.ndarray], title: str) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), dpi=180)
    if raw:
        a=np.asarray([r["point_xy_m"] for r in raw]); ax.scatter(a[:,0],a[:,1],s=2,c="#9ca3af",alpha=.08,label="raw ray-plane observations")
    if gt:
        for code,p in gt.items():
            s=.06; ax.add_patch(plt.Rectangle((p[0]-s/2,p[1]-s/2),s,s,fill=False,edgecolor="#2563eb",lw=.8,alpha=.7)); ax.text(p[0]+.03,p[1]+.03,str(code),color="#1d4ed8",fontsize=6)
    if pred:
        pp=np.asarray([x["point_xy_m"] for x in pred]); ax.scatter(pp[:,0],pp[:,1],s=38,c="#dc2626",edgecolors="white",lw=.5,zorder=5,label="filtered physical clusters")
        gg=np.asarray(list(gt.values())); ids=list(gt)
        for i,x in enumerate(pp):
            j=int(np.argmin(np.linalg.norm(gg-x,axis=1))); q=gg[j]; err=float(np.linalg.norm(x-q)); box=box_distance(x,q); ax.annotate(f"{pred[i]['cluster_id']}  center={err:.2f}m box={box:.2f}m",x,xytext=(4,4),textcoords="offset points",fontsize=7,color="#991b1b"); ax.annotate("",xy=q,xytext=x,arrowprops={"arrowstyle":"->","color":"#f97316","lw":.6,"alpha":.6})
    ax.set_aspect("equal",adjustable="datalim"); ax.grid(alpha=.2); ax.set_xlabel("nav/global X (m)"); ax.set_ylabel("nav/global Y (m)"); ax.set_title(title); ax.legend(loc="best"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run_variant(name: str, height: float, payload: dict[str, Any], frame_times: np.ndarray, nav_t: np.ndarray, nav_pose: np.ndarray, origin: np.ndarray, gt: dict[int,np.ndarray], out: Path, heading_offset: float, min_obs: int, gate: float, cluster_radius: float) -> dict[str, Any]:
    rows=raw_rows(payload,frame_times,nav_t,nav_pose,origin,height,heading_offset); aggs,rejected=robust_code_aggregates(rows,min_obs,gate); clusters=cluster_aggregates(aggs,cluster_radius); m=metrics(clusters,gt); render(out/f"markers_{name}.png",rows,clusters,gt,f"Notebook-style nav marker map — {name}, H={height:.2f} m, heading offset={heading_offset:.1f}°")
    report={"name":name,"plane_height_m":height,"heading_offset_deg":heading_offset,"raw_observations":len(rows),"codes":len(aggs),"physical_clusters":len(clusters),"per_code_aggregates":list(aggs.values()),"rejected_codes":rejected,"physical_clusters_data":clusters,"offline_gt_metrics":m}
    (out/f"markers_{name}.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); return report


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--recording",type=Path,default=DEFAULT_RECORDING); p.add_argument("--gt",type=Path,default=DEFAULT_GT); p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT); p.add_argument("--height",type=float,default=3.03); p.add_argument("--heading-offset-deg",type=float,default=0.0); p.add_argument("--min-observations",type=int,default=12); p.add_argument("--filter-gate-m",type=float,default=.30); p.add_argument("--cluster-radius-m",type=float,default=.30); args=p.parse_args()
    rec=args.recording; out=args.out_dir; out.mkdir(parents=True,exist_ok=True); payload=json.loads((rec/"observations.json").read_text()); frame_times=load_frame_times(rec/"frame_timestamps.csv",len(payload.get("data",[]))); nav_t,nav_pose,nav_meta=load_nav(next(rec.glob("fliftbot_move_logger_*.log"))); origin=load_camera_origin(rec/"camera_position.txt"); gt=load_gt(args.gt)
    fixed=run_variant("fixed_H303",args.height,payload,frame_times,nav_t,nav_pose,origin,gt,out,args.heading_offset_deg,args.min_observations,args.filter_gate_m,args.cluster_radius_m)
    auto=triangulate_height(raw_rows(payload,frame_times,nav_t,nav_pose,origin,args.height,args.heading_offset_deg),origin,args.min_observations,.20,args.heading_offset_deg)
    if auto.get("status")=="accepted":
        auto_report=run_variant("auto_height",float(auto["ceiling_height_m"]),payload,frame_times,nav_t,nav_pose,origin,gt,out,args.heading_offset_deg,args.min_observations,args.filter_gate_m,args.cluster_radius_m)
        auto_report["status"]="accepted"
        auto_report["height_diagnostic"]=auto
    else:
        # Remove stale products from an earlier run that happened to accept a
        # bad auto-height hypothesis.
        for stale in (out / "markers_auto_height.png", out / "markers_auto_height.json"):
            if stale.exists(): stale.unlink()
        auto_report={"name":"auto_height","status":"abstained","height_diagnostic":auto}
    report={"schema":"fliftbot.nav_marker_map_notebook_style.v1","runtime_method":"notebook ray-plane + nav_* interpolation","runtime_gt_used":False,"cartographer_used":False,"frame_fit_used":False,"inputs":{"recording":str(rec.resolve()),"observations":str((rec/"observations.json").resolve()),"frame_timestamps":str((rec/"frame_timestamps.csv").resolve()),"nav_source":nav_meta,"camera_position":str((rec/"camera_position.txt").resolve()),"gt_offline_only":str(args.gt.resolve())},"camera_origin_m":origin.tolist(),"geometry":{"formula":"P_local=O+((H-Oz)/dz)d; P_global=Rz(nav_dgr+heading_offset)P_local_xy+[nav_x,nav_y]","stored_ray_used_directly":True,"heading_offset_deg":args.heading_offset_deg,"height_fixed_m":args.height,"aggregation":"robust per-code filter, then physical clusters across decoder IDs","min_observations":args.min_observations,"filter_gate_m":args.filter_gate_m,"cluster_radius_m":args.cluster_radius_m},"fixed_H303":fixed,"auto_height":auto_report}
    (out/"comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"README.md").write_text(f"""# Notebook-style nav marker map\n\nOnly `nav_t/nav_x/nav_y/nav_dgr` from `{nav_meta['source']}` are used. `t/x/y/dgr`, encoder fields, Cartographer and GT are not runtime inputs. `nav_t` is parsed as UTC and interpolated to each `frame_timestamps.csv` timestamp.\n\nGeometry: `P_local=O+((H-Oz)/dz)d`, then `P_global=Rz(nav_dgr+{args.heading_offset_deg:.1f}°)P_local_xy+[nav_x,nav_y]`; origin is `{origin.tolist()}` m from `camera_position.txt`, and stored `item['ray']` is used directly.\n\nThe primary run is `markers_fixed_H303.png` (`H=3.03 m`). `markers_auto_height.png` is emitted only if a map-independent multi-view height estimate is consistent; otherwise `auto_height` is explicitly abstained in `comparison.json`. Red points are robust physical clusters (decoder IDs may be merged), gray points are raw observations, blue squares are offline GT boxes, orange arrows show nearest-GT displacement. Labels show `center` error to the nearest GT centre and `box` distance to the nearest 6 cm GT box. GT is plotted in the source encoder-tag frame (`encoder_tag_cm/100`), with no rotation/scale/fit.\n\n""",encoding="utf-8")
    print(json.dumps({"out":str(out.resolve()),"fixed_clusters":len(fixed["physical_clusters_data"]),"auto":auto_report.get("status", "abstained")},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
