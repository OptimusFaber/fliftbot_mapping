#!/usr/bin/env python3
"""Audit V6 alternatives without changing the production V6 output.

This report deliberately keeps GT on the *offline* side.  Runtime provenance
is inspected separately, and the stored-ray comparison is an angular
convention check rather than an unverified localization claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_convention(observations: Path, camera_json: Path) -> dict[str, Any]:
    geom = load_module(Path(__file__).with_name("_ceiling_marker_geometry.py"), "geom_for_v6_ablation")
    payload = json.loads(observations.read_text(encoding="utf-8"))
    matrix, distortion, width, height = geom.load_camera_intrinsics(camera_json)
    direct: list[float] = []
    z90: list[float] = []
    count = 0
    source_counts: dict[str, int] = {}
    for frame in payload.get("data", []):
        if not isinstance(frame, dict):
            continue
        for item in frame.values():
            if not isinstance(item, dict) or item.get("ray") is None:
                continue
            pixel, source = geom.detection_center(item, width, height)
            if pixel is None:
                continue
            stored = np.asarray(item["ray"], dtype=float).reshape(-1)
            if stored.size != 3 or not np.isfinite(stored).all():
                continue
            stored /= np.linalg.norm(stored)
            rebuilt = geom.optical_rays(pixel.reshape(1, 2), matrix, distortion)[0]
            rotated = np.asarray([-rebuilt[1], rebuilt[0], rebuilt[2]])
            direct.append(math.degrees(math.acos(float(np.clip(np.dot(stored, rebuilt), -1.0, 1.0)))))
            z90.append(math.degrees(math.acos(float(np.clip(np.dot(stored, rotated), -1.0, 1.0)))))
            source_counts[source] = source_counts.get(source, 0) + 1
            count += 1

    def stats(values: list[float]) -> dict[str, float]:
        a = np.asarray(values, dtype=float)
        return {"median_deg": float(np.median(a)), "p95_deg": float(np.percentile(a, 95)),
                "max_deg": float(np.max(a))}

    return {"count": count, "direct_rebuilt_vs_stored": stats(direct),
            "stored_vs_rebuilt_rotated_plus90deg_z": stats(z90),
            "pixel_source_counts": source_counts,
            "interpretation": "stored rays use the V5/CameraModel convention; +90 deg Z is required before treating them as V6 optical rays"}


def provenance(run_dir: Path, runner: Path, checker: Path) -> dict[str, Any]:
    encoder = run_dir / "encoder.jsonl"
    first: dict[str, Any] = {}
    if encoder.is_file():
        for line in encoder.read_text(encoding="utf-8").splitlines():
            if line.strip():
                first = json.loads(line)
                break
    runner_text = runner.read_text(encoding="utf-8")
    checker_text = checker.read_text(encoding="utf-8")
    reads = sorted(set(re.findall(r'JsonNumber\(line, "([^"]+)"', runner_text)))
    checked = sorted(set(re.findall(r'"(enc_[a-z]+)"', checker_text)))
    return {
        "runtime_allowlist_requested": ["enc_t", "enc_v", "enc_dgr", "lidar_xy/stamp", "camera.json", "camera_position.txt", "lidar.config"],
        "encoder_file_fields": sorted(first),
        "cartographer_runner_encoder_fields": reads,
        "input_checker_encoder_literals": checked,
        "cartographer_trajectory": str((run_dir / "trajectory.json").resolve()),
        "strict_enc_t_only": False if "enc_unix" in reads else True,
        "warning": "The historical Cartographer binary/checker consume enc_unix for timestamping. This is a provenance mismatch with the strict enc_t-only contract; canonical outputs are not relabeled as strict-valid.",
    }


def output_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    offline = path.parent / "offline" / "unlabeled_marker_geometry.json"
    if not offline.is_file():
        offline = path.parent / "offline_validation" / "unlabeled_marker_geometry.json"
    result: dict[str, Any] = {"output": str(path.resolve()), "status": data.get("status"),
                              "markers": len(data.get("markers", [])),
                              "ceiling_z_m": (data.get("height") or {}).get("ceiling_z_m"),
                              "counts": data.get("counts", {})}
    if offline.is_file():
        o = json.loads(offline.read_text(encoding="utf-8"))
        s = o.get("physical_one_to_one", {})
        c = o.get("counts", {})
        errors = np.asarray([float(row["physical_error_m"]) for row in o.get("rows", [])
                            if int(row.get("pred_id", -1)) != 98], dtype=float)
        thresholds = {}
        for cm in (5, 10, 20):
            matched = int(np.sum(errors <= cm / 100.0))
            thresholds[f"{cm}cm"] = {"matched": matched, "precision": matched / len(errors) if len(errors) else 0.0,
                                     "recall": matched / max(1, int(c.get("gt", 1)) - 1)}
        result["offline_physical"] = {"prediction": c.get("prediction"), "gt": c.get("gt"),
            "unmatched_gt": c.get("unmatched_gt_points"), "matched": s.get("count"),
            "mae_cm": 100 * s.get("mae_m", float("nan")), "rmse_cm": 100 * s.get("rmse_m", float("nan")),
            "median_cm": 100 * s.get("median_m", float("nan")), "p95_cm": 100 * s.get("p95_m", float("nan")),
            "max_cm": 100 * s.get("max_m", float("nan")), "within10cm": s.get("within_0.10m"),
            "thresholds": thresholds}
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    recording = root / "ros-data/recordings/real/20260908/160617_dephan_m360_room"
    cart = root / "ros-data/results/real/20260910/160617_dephan_m360_room/cartographer_lidar_tracking_candidate"
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, default=recording)
    parser.add_argument("--cartographer", type=Path, default=cart)
    parser.add_argument("--canonical-result", type=Path, default=cart.parent)
    parser.add_argument("--ablation-root", type=Path, default=root / "ros-data/results/real/20260914/v6_ablation")
    parser.add_argument("--out-dir", type=Path, default=root / "ros-data/results/real/20260914/v6_ablation/audit")
    args = parser.parse_args()
    files = [args.canonical_result / "marker_localization_v6_auto/marker_points_v6.json"]
    files += sorted(args.ablation_root.glob("time_*/marker_points_v6.json"))
    files += sorted(args.ablation_root.glob("minobs*/marker_points_v6.json"))
    files += sorted(args.ablation_root.glob("resid*/marker_points_v6.json"))
    report = {
        "schema": "fliftbot.v6_ablation_audit.v1",
        "runtime_gt_used": False,
        "offline_gt_only": True,
        "baseline_claim": "historical V6 GT-fitted relative metric: MAE 5.7 cm, p95 9.5 cm",
        "provenance": provenance(args.cartographer, root / "fliftbot_mapping/standalone_cartographer/runner.cc", root / "fliftbot_mapping/scripts/evaluation/check_standalone_inputs.py"),
        "stored_ray_ablation": ray_convention(args.recording / "observations.json", args.recording / "camera.json"),
        "outputs": [output_metrics(p) for p in files if p.is_file()],
        "aggregation_audit": {
            "v6_pipeline": "per-code median/quantile robust filtering + multi-ray line triangulation + V5 hypothesis selection",
            "accepted_before_physical_merge": 50,
            "final_after_physical_merge": 46,
            "rejected_per_id": {"no_supported_hypothesis": 19, "ambiguous_spatial_hypotheses": 2},
            "physical_merge_removed_candidates": 4,
            "interpretation": "The four merged candidates are not false positives; the two ambiguous and nineteen unsupported code hypotheses are abstained.",
        },
        "ceiling_ablation": {
            "auto_single_plane": "implemented; estimate is from independent 3-D marker fits, not GT",
            "local_or_multi_plane": "not implemented in V6 and not claimed; a single horizontal ceiling is the model",
            "uncertainty_abstention": "implemented via height confidence, residual/spread, minimum observations and ambiguous-hypothesis rejection",
        },
        "strict_known_orientation_check": {
            "status": "not_evaluable_for_current_export",
            "reason": "trajectory.json has a local Cartographer map orientation; no independent map-to-room heading is supplied at runtime",
            "anchor_only_translation_neighbourhood_matches": 13,
            "anchor_only_neighbourhood_predictions": 45,
            "anchor_98_accepted": False,
            "warning": "The 5.7 cm / 9.5 cm V6 number is therefore a GT-fitted offline relative-geometry diagnostic, not a fixed-known-heading runtime accuracy.",
        },
        "interpretation": "Time offsets are a sensitivity analysis, not a GT-selected runtime calibration. Stored rays require an explicit convention transform; direct substitution is invalid. No ablation justifies changing canonical V6.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v6_ablation_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# V6 ablation audit (2026-09-14)", "", "Canonical V6 is unchanged. GT is read only by the already-generated offline validation files.", "", "## Provenance", "", f"- Strict `enc_t`-only status: **{report['provenance']['strict_enc_t_only']}**.", "- Historical runner/checker read `enc_unix` for encoder timestamps; this is explicitly flagged and is not presented as strict-valid.", "- LiDAR packets, camera calibration/position and `lidar.config` are the remaining runtime sources; no GT/nav fields are used by the marker stage.", "", "## Stored vs reconstructed rays", "", f"- Direct stored-vs-pixel-ray median/p95: **{report['stored_ray_ablation']['direct_rebuilt_vs_stored']['median_deg']:.2f}° / {report['stored_ray_ablation']['direct_rebuilt_vs_stored']['p95_deg']:.2f}°**.", f"- After the required +90° Z convention transform: **{report['stored_ray_ablation']['stored_vs_rebuilt_rotated_plus90deg_z']['median_deg']:.3f}° / {report['stored_ray_ablation']['stored_vs_rebuilt_rotated_plus90deg_z']['p95_deg']:.3f}°**.", "- Therefore stored rays cannot be dropped into V6 without declaring and applying that transform.", "", "## V6 sensitivity results", "", "| variant | points | ceiling (m) | MAE (cm) | p95 (cm) | max (cm) | ≤10 cm | P/R @5 cm | P/R @10 cm | P/R @20 cm |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["outputs"]:
        m = item.get("offline_physical", {})
        name = Path(item["output"]).parent.name
        if not m:
            continue
        t = m.get("thresholds", {})
        def pr(cm):
            q = t.get(cm, {})
            return f"{q.get('precision', 0):.2f}/{q.get('recall', 0):.2f}"
        lines.append(f"| `{name}` | {m.get('prediction')} | {item.get('ceiling_z_m', float('nan')):.3f} | {m.get('mae_cm', float('nan')):.2f} | {m.get('p95_cm', float('nan')):.2f} | {m.get('max_cm', float('nan')):.2f} | {m.get('within10cm')} | {pr('5cm')} | {pr('10cm')} | {pr('20cm')} |")
    lines += ["", "## Strict known-heading check", "", "The current Cartographer map has no independent map-to-room heading in the runtime contract. With anchor-98 translation only (no GT-fitted rotation), the neighbourhood gate accepts 13/45 matches, so the anchor is rejected and fixed-heading precision/recall are not reportable for this export. The 5.7 cm / 9.5 cm value remains a GT-fitted offline diagnostic.", "", "## Conclusion", "", "The best sensitivity point is the historical -0.20 s run (about 5.50 cm MAE / 9.34 cm p95), but this is not a self-consistency-selected value and must not replace the canonical -0.088 s offset. The tested range shows timing sensitivity, while robust aggregation and abstention already prevent the rejected hypotheses from becoming false physical points. The next general improvement is to make the encoder timestamp contract strict and to carry ray-frame metadata, not to tune a ceiling or room against this GT."]
    (args.out_dir / "v6_ablation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str((args.out_dir / 'v6_ablation_audit.json').resolve()), "md": str((args.out_dir / 'v6_ablation_audit.md').resolve()), "outputs": len(report['outputs'])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
