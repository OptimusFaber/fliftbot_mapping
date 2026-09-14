"""Uncompressed .npz lidar frames, same layout as LiDAR-data/log_pandar40p_frames.py.

Keys written per horizon frame (mapping reads these):
  xy (N,2) float32, xyz (N,3) float32, intensity (N,) float32, t (N,) float64,
  stamp (), frame (), channel ()
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

# sensor_msgs/PointField datatype → numpy
_PF_NP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def enc_time_str(unix_t: float) -> str:
    dt = datetime.fromtimestamp(unix_t, tz=timezone.utc)
    ms = int((unix_t - int(unix_t)) * 1000)
    return dt.strftime('%Y_%m_%d_%H_%M_%S_') + f'{ms:03d}'


def safe_ts_name(ts_str: str) -> str:
    return ts_str.replace(':', '-').replace('+00:00', '+00-00')


def horizon_npz_arrays(
    xy,
    xyz,
    intensity,
    times,
    stamp: float,
    frame: int,
    channel: int,
) -> dict[str, np.ndarray]:
    return {
        'xy': np.ascontiguousarray(xy, dtype=np.float32),
        'xyz': np.ascontiguousarray(xyz, dtype=np.float32),
        'intensity': np.ascontiguousarray(intensity, dtype=np.float32),
        't': np.ascontiguousarray(times, dtype=np.float64),
        'stamp': np.asarray(stamp, dtype=np.float64),
        'frame': np.asarray(frame, dtype=np.int32),
        'channel': np.asarray(channel, dtype=np.int32),
    }


class AsyncNpzWriter:
    """Background uncompressed .npz writer (does not block the ROS callback)."""

    def __init__(self, max_queue: int = 64):
        self._queue: Queue = Queue(maxsize=max_queue)
        self._saved = 0
        self._dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, path: Path, arrays: dict[str, np.ndarray]) -> bool:
        try:
            self._queue.put_nowait((path, arrays))
            return True
        except Full:
            self._dropped += 1
            return False

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                path, arrays = self._queue.get(timeout=0.2)
            except Empty:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
            self._saved += 1
            self._queue.task_done()

    def close(self, timeout: float = 30.0):
        self._stop.set()
        self._thread.join(timeout=timeout)
        return {
            'saved': self._saved,
            'dropped': self._dropped,
            'pending': self._queue.qsize(),
        }


def cloud_as_numpy(msg) -> np.ndarray:
    """Zero-copy structured view of sensor_msgs/PointCloud2 (little-endian)."""
    names, formats, offsets = [], [], []
    endian = '>' if msg.is_bigendian else '<'
    for field in msg.fields:
        np_t = _PF_NP.get(int(field.datatype))
        if np_t is None:
            continue
        names.append(field.name)
        dt = np.dtype(np_t).newbyteorder(endian)
        count = int(getattr(field, 'count', 1) or 1)
        formats.append(dt if count <= 1 else np.dtype((dt, (count,))))
        offsets.append(int(field.offset))
    dtype = np.dtype({
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': int(msg.point_step),
    })
    n = int(msg.width) * int(msg.height)
    buf = msg.data
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        buf = bytes(buf)
    return np.frombuffer(buf, dtype=dtype, count=n)


def _point_times(n: int, stamp_t: float, scan_period: float = 0.1) -> np.ndarray:
    if n <= 1:
        return np.full(n, stamp_t, dtype=np.float64)
    dt = float(scan_period) / float(n)
    return stamp_t + np.arange(n, dtype=np.float64) * dt


def _pack_horizon(xs, ys, zs, intensity, stamp_t, channel: int, scan_period=0.1):
    n = int(xs.shape[0])
    xy = np.column_stack((xs, ys))
    xyz = np.column_stack((xs, ys, zs))
    if intensity is None:
        inten = np.zeros(n, dtype=np.float32)
    else:
        inten = np.asarray(intensity, dtype=np.float32)
    times = _point_times(n, stamp_t, scan_period)
    return xy, xyz, inten, times, int(channel)


def extract_horizon_from_xyz(
    x,
    y,
    z,
    *,
    intensity=None,
    rings=None,
    height: int = 1,
    width: int = 0,
    stamp_t: float = 0.0,
    min_range: float = 0.3,
    max_range: float = 200.0,
    scan_period: float = 0.1,
):
    """Pick the ~0° elevation ring from a 3D cloud (or the whole 2D scan).

    Gazebo gpu_lidar is organized as height=vertical × width=horizontal and
    often has no `ring` field. Real Pandar uses ring; we accept both.

    Channel 13 is *not* used for Gazebo: real Pandar spacing is uneven, sim
    vertical samples are uniform −25°…+15°, so ring 12 is ~−13°, not horizon.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    rxy = np.hypot(x, y)
    finite = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & (rxy >= float(min_range))
        & (rxy <= float(max_range))
    )
    elev = np.degrees(np.arctan2(z, np.maximum(rxy, 1e-6)))
    inten_all = None if intensity is None else np.asarray(intensity).reshape(-1)

    n = int(x.size)
    height = int(height) if height else 1
    width = int(width) if width else (n // height if height else n)

    def _take(mask, channel):
        xs, ys, zs = x[mask], y[mask], z[mask]
        inten = None if inten_all is None else inten_all[mask]
        return _pack_horizon(xs, ys, zs, inten, stamp_t, channel, scan_period)

    if rings is not None:
        rings = np.asarray(rings).reshape(-1)
        valid = finite & (rings >= 0)
        if valid.any():
            best_ring = None
            best_abs = float('inf')
            for ring in np.unique(rings[valid]):
                sel = valid & (rings == ring)
                if int(sel.sum()) < 8:
                    continue
                med = float(np.median(elev[sel]))
                if abs(med) < best_abs:
                    best_abs = abs(med)
                    best_ring = int(ring)
            if best_ring is not None:
                # 0-based ring → 1-based channel (Pandar style)
                return _take(valid & (rings == best_ring), best_ring + 1)

    organized = height > 1 and width > 1 and n == height * width
    if organized:
        elev_r = elev.reshape(height, width)
        fin_r = finite.reshape(height, width)
        best_row = 0
        best_abs = float('inf')
        for row in range(height):
            sel = fin_r[row]
            if int(sel.sum()) < 8:
                continue
            med = float(np.median(elev_r[row][sel]))
            if abs(med) < best_abs:
                best_abs = abs(med)
                best_row = row
        row_mask = np.zeros(n, dtype=bool)
        row_mask[best_row * width:(best_row + 1) * width] = True
        return _take(finite & row_mask, best_row + 1)

    # 2D / unstructured: keep near-horizon, or everything if the scan is flat
    if finite.any() and float(np.nanmax(np.abs(elev[finite]))) <= 2.5:
        return _take(finite, 0)
    near = finite & (np.abs(elev) <= 1.25)
    if int(near.sum()) >= 8:
        return _take(near, 0)
    return _take(finite, 0)


def laserscan_to_horizon(
    ranges,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    stamp_t: float,
    time_increment: float = 0.0,
    scan_time: float = 0.0,
    min_keep: float = 0.15,
):
    """Convert a 2D LaserScan into horizon npz arrays (sensor XY, z=0)."""
    ranges = np.asarray(ranges, dtype=np.float64)
    n = int(ranges.size)
    angles = float(angle_min) + np.arange(n, dtype=np.float64) * float(angle_increment)
    rmin = max(float(range_min), float(min_keep))
    rmax = float(range_max)
    keep = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
    r = ranges[keep]
    a = angles[keep]
    xs = (r * np.cos(a)).astype(np.float32)
    ys = (r * np.sin(a)).astype(np.float32)
    zs = np.zeros_like(xs)
    period = float(scan_time or 0.0)
    if period <= 0.0 and n > 1:
        period = float(time_increment or 0.0) * float(n)
    if period <= 0.0:
        period = 0.1
    idx = np.nonzero(keep)[0]
    times = (float(stamp_t) + idx.astype(np.float64) / float(n) * period
             if n > 1 else np.full(idx.size, float(stamp_t)))
    xy = np.column_stack((xs, ys))
    xyz = np.column_stack((xs, ys, zs))
    intensity = np.zeros(xs.shape[0], dtype=np.float32)
    return xy, xyz, intensity, times, 0
