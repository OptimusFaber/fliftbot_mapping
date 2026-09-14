# Ceiling-marker runtime audit — 2026-09-11

## Result

The current DePHAN pass supports a scene-independent unknown-height mode when
the trajectory frame contract is honoured. V6 estimated `ceiling_z=3.0458 m`
from 50 independent marker hypotheses (2 rejected), with robust sigma `6.5 cm`
and confidence `0.52`. It produced 46 points. GT was not read by V6.

Offline only, after centering on marker 98 and fitting one rotation (no scale,
translation, or runtime registration), V6 scored:

| metric | V6 auto-height |
|---|---:|
| physical points | 46 (45 non-anchor) |
| rotation to GT | 43.021 deg |
| MAE | 5.67 cm |
| RMSE | 6.43 cm |
| median | 5.38 cm |
| p95 | 9.49 cm |
| maximum | 9.63 cm |
| within 10 cm | 45/45 |

This is effectively the same as the best fixed 3.03 m result (MAE 5.68 cm,
RMSE 6.45 cm, p95 9.52 cm). The auto estimate is therefore not silently
degrading this recording.

## Root cause found in the previous auto experiment

The old `overlay_marker_rays_on_cartographer_map.py --height-mode auto` path
used the camera position relative to the rotation axis while consuming a
trajectory whose `tracking_frame` was `lidar`. Its camera origin was therefore
wrong by the LiDAR lever arm. Pairwise ray triangulation then returned arbitrary
heights (roughly 1.2–2.0 m in the diagnostic run) instead of the physical
ceiling. This was a frame-contract error, not evidence that unknown height is
unobservable.

V6 reads `trajectory.json.tracking_frame`, reads `lidar.config`, and for a LiDAR
trajectory subtracts the LiDAR XY mount from the camera XY mount exactly once.
For a rotation-axis trajectory it performs no subtraction. The corrected
DePHAN offset is camera-from-LiDAR `[-0.165, +0.600, +1.270] m` under the
recording's documented lateral-sign conversion.

## Error character

The 303 cm and 313 cm fixed-height outputs share 46 physical points. Increasing
the plane by 10 cm changes their XY by only 3.3 cm mean (3.2 cm median, 6.0 cm
p95, 6.9 cm maximum), with mean bias only `[+0.09,+0.17] cm`. Thus the height
error appears primarily as a radial scale/depth effect, not a global XY
translation. The relative point error is approximately 5–6 cm MAE and does not
grow materially from 0–4 m radial distance in this pass (about 5.2 cm for
0–2 m and 5.9 cm for 2–4 m). This is consistent with a mixture of camera ray,
trajectory timing and map error rather than one large lever-arm offset.

The points decoded as IDs 274, 629 and 1982 are physical matches despite their
semantic ID mismatch. The ID-independent matcher finds them within the GT
point set; this is a decoder/track identity ambiguity, not a metre-scale
geometric displacement.

## What V6 does and does not estimate

V6 estimates only the ceiling plane height from repeated observations. It keeps
the installation camera angle and clock offset explicit, because jointly fitting
height, camera roll/pitch/yaw, time offset and trajectory error from one weak
planar pass is underdetermined and can absorb one error into another. If fewer
than three independent robust 3-D marker fits agree, V6 returns
`status=abstained` and does not emit fabricated points.

The runtime cannot infer a unique plane height from a stationary camera or
nearly parallel rays. A useful pass needs parallax: each marker should be seen
from multiple camera positions with nontrivial baseline. Time-offset estimation
also needs motion with observable angular/linear change and should be enabled
only as a bounded diagnostic/calibration stage.

## Portability

DePHAN, Pandar and RPLIDAR are supported by the same image stage if their
Cartographer export supplies metric timestamped trajectory poses and declares
the tracking frame. The recording must also contain `camera.json`,
`camera_position.txt`, frame timestamps and either `lidar.config` or an explicit
rotation-axis trajectory declaration. Per-point LiDAR timing is needed when
building the Cartographer map, but not when intersecting camera rays after the
trajectory is exported. Missing or inconsistent timing should lower confidence
or cause abstention, not trigger scene-specific constants.

## Files

- Runtime candidate: `scripts/evaluation/ceiling_marker_localization_v6.py`
- Candidate design: `docs/ceiling_marker_localization_v6.md`
- Tests: `tests/test_ceiling_marker_localization_v6.py`
- DePHAN candidate JSON from the validation run: `/tmp/luna_v6.HaJVd1/marker_points_v6.json`
- Offline V6 comparison: `/tmp/luna_v6_eval.825QPC/unlabeled_marker_geometry.json`

The `/tmp` paths are diagnostic artifacts only; rerun the CLI with a persistent
`--out-dir` to retain them in `ros-data/results` without touching canonical V5
outputs.
