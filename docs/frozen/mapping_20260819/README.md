# Frozen mapping baseline — 2026-08-19

This directory is the immutable working baseline for the reproduced warehouse
experiment `maps/20260815/sim/run_004137`.

## Contents

- `mapping/`: exact copy of `/home/optimus/ros-projects/archieve/src/mapping/*.py`.
- `output/` and `runtime/`: exact local dependencies required by the frozen mapper.
- `run_archive_all_methods.py`: runner used to reproduce `clouds`, `odom`,
  `improved`, and `factor`.
- `overlay/overlay_archive_warehouse_gt.py`: fixed GT overlay implementation.
- `SHA256SUMS`: integrity hashes for every frozen source file.

## Frozen method definitions

The authoritative presets are `METHOD_PRESETS` in
`mapping/config.py`. The frozen runner imports this local snapshot and must not substitute modules from the
current `src/flifltbot` tree.

- `clouds`: LiDAR-only trajectory, no encoder, no deskew.
- `odom`: encoder trajectory, no deskew.
- `improved`: encoder + full SE(2) deskew + synthetic point timestamps; turn
  frames are filtered from the stable hit-grid.
- `factor`: encoder priors + sequential LiDAR/submap constraints + loop
  closures + full SE(2) deskew.

## Frozen GT overlay contract

1. Synchronize predicted body poses and Gazebo `/odom` body poses by time.
2. Estimate one rigid SE(2) transform from predicted body trajectory to GT
   body trajectory. Scale and reflection are forbidden.
3. Apply a small rigid yaw/translation refinement using the four outer GT wall
   bands. The same final transform is applied to the predicted map and route.
4. The archived horizon NPZ poses are already poses of the historical upper
   LiDAR at `x=0.95 m`, not body poses. Project prediction to the approved
   `x=1.14 m` point with only `R(yaw) * [0.19, 0]`. Project Gazebo `/odom`
   body GT with `R(yaw) * [1.14, 0]`. Adding `1.14 m` to prediction is a
   double lever-arm application and incorrectly enlarges turning arcs.
5. Draw warehouse GT outer walls, all shelves, and all structural columns.
6. Save each method's aligned coordinates to `*_aligned_data.npz`. The combined
   comparison PNG is a literal collage of the individual PNGs, so route
   geometry cannot differ because of a second rendering pass.

The historical 2026-08-15 Gazebo model and archived mapper used an upper-LiDAR
lever of `0.95 m`. The current approved geometry is `1.14 m`; therefore route
comparison deliberately evaluates both GT and prediction at `1.14 m`. This
historical mismatch is recorded in the overlay result `summary.json`.

Expected outputs:

- `results/20260819/archive_all_methods_reproduced/warehouse/`
- `results/20260819/archive_all_methods_reproduced/warehouse/gt_lidar_corner/`

Reproduce:

```bash
PYTHONPATH=src python3 frozen/mapping_20260819/run_archive_all_methods.py \
  --scene warehouse \
  --out-root results/20260819/archive_all_methods_reproduced

MPLCONFIGDIR=/tmp/mpl-overlay PYTHONPATH=src \
  python3 scripts/evaluation/experiments/overlay_archive_warehouse_gt.py
```
