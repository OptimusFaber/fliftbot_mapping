# V6 ceiling-marker localization

V6 is a candidate runtime method for new scenes. It preserves the validated V5
pixel/ray convention and does not read GT, room dimensions, or use an SE(2)
registration. The important change is that the ceiling height is not a runtime
input: repeated observations of each decoded code are triangulated as 3-D rays,
then a robust consensus of their Z coordinates estimates one horizontal ceiling
plane. Points are finally reconstructed with the existing robust V5 hypothesis
selection.

## Frame contract

`camera_position.txt` is centimetres relative to the rotation axis. Its lateral
sign is converted once using `--camera-y-sign`. If `trajectory.json` has
`tracking_frame=lidar`, `lidar.config` is read and the LiDAR position is
subtracted once, producing the camera offset relative to the Cartographer pose.
For a rotation-axis trajectory, no subtraction is made. This distinction is
critical: triangulation with the camera origin at the axis while the trajectory
is in LiDAR frame produces arbitrary Z estimates and a distorted map.

## Example

```bash
python3 scripts/evaluation/ceiling_marker_localization_v6.py \
  --observations /path/recording/observations.json \
  --trajectory /path/cartographer/trajectory.json \
  --camera-json /path/recording/camera.json \
  --camera-position /path/recording/camera_position.txt \
  --lidar-config /path/recording/lidar.config \
  --trajectory-pose-frame auto \
  --out-dir /path/result/marker_localization_v6_auto
```

The output `marker_points_v6.json` contains `status`, estimated `ceiling_z_m`,
robust sigma, confidence, frame conversion metadata, per-code 3-D triangulation
fits, and final marker points. `status=abstained` is a valid result: it means the
recording did not provide enough independent parallax or the rays were
inconsistent, and no fabricated marker positions are emitted.

## Observability and limitations

Unknown ceiling height is observable only when a marker is seen from sufficiently
different camera positions/angles. A stationary camera, nearly parallel rays,
wrong trajectory frame, or unmodelled time offset makes height and extrinsic
corrections underdetermined. V6 therefore keeps the installation camera angles
and clock offset as explicit metadata and abstains when independent 3-D fits do
not agree. Automatic time-offset/extrinsic estimation is a separate calibration
stage; fitting those parameters jointly from one weak pass can silently absorb
trajectory error and must not be enabled by default.

The method is portable to DePHAN, Pandar and RPLIDAR as long as each map export
contains metric `trajectory.json` poses with a declared tracking frame and the
recording exports `camera.json`, `camera_position.txt`, frame timestamps and a
sensor `lidar.config` (or declares a rotation-axis pose). Per-point LiDAR timing
is needed by Cartographer map generation, but not by the image-ray stage once the
trajectory is trustworthy.
