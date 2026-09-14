# Frozen warehouse baseline — 2026-08-20

This manifest freezes the best non-oracle replay found for
`maps/20260820/sim/run_184526`.  The result itself remains in `results/` to
avoid duplicating the large point-cloud JSON.

## Approved baseline

- variant: `deskew_wheel`
- trajectory: wheel odometry + sequential LiDAR-to-submap factors + loops
- deskew: wheel motion only; no synthetic per-beam timestamps
- rear rotation pivot: `(0, 0)`
- recorded upper-LiDAR lever: `x=1.14 m`, `z=1.645 m`
- submap: 10 keyframes
- map grid: 0.05 m, minimum 5 hits
- GT alignment: one rigid SE(2), no scale or reflection; trajectory evaluated
  at the LiDAR point

Authoritative artifacts:

- `results/20260820/sim/run_184526/test/full/deskew_wheel/result.json`
- `results/20260820/sim/run_184526/test/full/deskew_wheel/summary.json`
- `results/20260820/sim/run_184526/test/full/deskew_wheel/result_vs_gt.png`
- `results/20260820/sim/run_184526/test/full/deskew_wheel/result_gt_alignment.json`
- full comparison: `results/20260820/sim/run_184526/test/SUMMARY.md`

Expected metrics:

- ATE RMSE: `0.3960877686 m` (contains one 17.29 m GT-sync outlier)
- median: `0.1464033210 m`
- p95: `0.3228469231 m`
- turn RMSE: `0.1642936745 m`
- turn median: `0.1320849445 m`
- yaw RMSE: `1.9737464111 deg`

The Gazebo `/odom` oracle is a diagnostic reference, not a mapping baseline.
Its median/p95 are 0.0245/0.0537 m, proving that the remaining error is mainly
trajectory/timing related rather than map rasterization.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/mpl_deskew_wheel PYTHONPATH=src \
python3 scripts/evaluation/experiments/exp_sim_restore_variant.py deskew_wheel \
  --run maps/20260820/sim/run_184526 \
  --out-root results/20260820/sim/run_184526/test/full

MPLCONFIGDIR=/tmp/overlay_deskew_wheel PYTHONPATH=src \
python3 scripts/evaluation/rebuild_all_gt_overlays.py --scene sim \
  --map results/20260820/sim/run_184526/test/full/deskew_wheel/result.json
```

## Development rule

Future experiments must write to a new directory and compare against this
baseline.  Do not overwrite these four authoritative artifacts.  Promote a
new baseline only if wall geometry improves and median, p95, turn error and yaw
error do not regress.

