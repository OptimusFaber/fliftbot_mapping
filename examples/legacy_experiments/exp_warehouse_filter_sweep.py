#!/usr/bin/env python3
"""One expensive factor replay, then cheap map-insertion ablations."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from flifltbot.mapping.map_filtering import build_global_map_filtered
from flifltbot.mapping.pipeline import sew_all_local_frames


def main():
    run = ROOT / "maps/20260820/sim/run_184526"
    out = ROOT / "results/20260820/sim/run_184526/test/full/filters"
    out.mkdir(parents=True, exist_ok=True)
    _p, records, _files, base_stats, frames, _raw_poses = sew_all_local_frames(
        run, use_encoder=True, cameras=["up"], source="sim", up_body_x=1.14,
        deskew_mode="full_se2", synthesize_beam_times=True, factor_mode=True,
        frontend_icp="submap", lidar_seq_target="submap", submap_keyframes=10,
        skip_turn_wz=None, map_min_hits=1, return_internals=True,
    )

    variants = []
    variants += [(f"wz_{x}", {"wz": None if x == "none" else float(x)})
                 for x in ("none", .05, .10, .15, .20, .30)]
    variants += [(f"steer_{x}", {"steer": float(x)})
                 for x in (10, 20, 30, 45, 60)]
    variants += [(f"sector_{x}", {"sector": float(x)})
                 for x in (360, 270, 180, 120)]
    variants += [(f"hits_{x}", {"hits": int(x)}) for x in (2, 3, 5, 8, 10)]
    variants += [(f"cell_{str(x).replace('.', '')}", {"cell": float(x)})
                 for x in (.03, .05, .10)]
    variants += [(f"component_{x}", {"component": int(x)})
                 for x in (10, 30, 50, 100)]

    summaries = []
    for name, cfg in variants:
        selected = list(frames)
        if "steer" in cfg:
            selected = [
                np.empty((0, 2)) if abs(float((r.get("synced_encoder") or {}).get("enc_dgr", 0))) > cfg["steer"] else p
                for p, r in zip(frames, records)
            ]
        if cfg.get("sector", 360) < 360:
            half = math.radians(cfg["sector"] / 2)
            masked = []
            for pts, rec in zip(selected, records):
                enc = rec.get("synced_encoder") or {}
                heading = math.radians(float(enc.get("enc_dgr", 0)))
                if float(enc.get("enc_v", 0)) < 0:
                    heading += math.pi
                a = np.arctan2(pts[:, 1], pts[:, 0]) if len(pts) else np.array([])
                d = np.arctan2(np.sin(a-heading), np.cos(a-heading))
                masked.append(pts[np.abs(d) <= half])
            selected = masked
        cell = cfg.get("cell", .05)
        points, transformed, raw, stats = build_global_map_filtered(
            selected, _raw_poses, encoders=[r.get("synced_encoder") for r in records],
            frame_times=[r.get("t") for r in records],
            skip_turn_wz=cfg.get("wz", .15), min_hits=cfg.get("hits", 5),
            cell_m=cell, min_component_cells=cfg.get("component", 50),
        )
        payload = {"variant": name, "config": cfg, "map": points,
                   "poses": records, "n_frames": len(records), "cell_m": cell,
                   "stats": stats, "base_stats": base_stats}
        (out / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
        arr = np.asarray(points)
        fig, ax = plt.subplots(figsize=(9, 9))
        if len(arr): ax.scatter(arr[:, 0], arr[:, 1], s=1)
        traj = np.asarray([r["pose"] for r in records])
        ax.plot(traj[:, 0], traj[:, 1], color="red", linewidth=.7)
        ax.set_aspect("equal"); ax.grid(alpha=.2); ax.set_title(name)
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=130); plt.close(fig)
        summaries.append({"variant": name, "config": cfg, "cells": len(points), **stats})
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
