"""Read and validate the fliftbot_map_build -> LiDAR navigation contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


def read_reference(path: Path) -> tuple[float, float, float]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("frame") != "cartographer_map":
        raise ValueError(f"{path}: reference frame must be cartographer_map")
    xy = reference.get("origin_map_xy_m", reference.get("marker98_map_xy_m"))
    if xy is None or len(xy) != 2:
        raise ValueError(f"{path}: missing origin_map_xy_m/marker98_map_xy_m")
    yaw = reference.get("origin_map_yaw_rad", reference.get("marker98_map_yaw_rad", 0.0))
    result = (float(xy[0]), float(xy[1]), float(yaw))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{path}: non-finite reference")
    return result


def read_y_axis_sign(path: Path) -> int:
    """Handedness of published navigation coordinates; ROS/map stays unchanged."""
    value = json.loads(path.read_text(encoding="utf-8")).get("y_axis_sign", 1)
    if value not in (-1, 1):
        raise ValueError(f"{path}: y_axis_sign must be -1 or +1")
    return int(value)


def validate_bundle(directory: Path) -> dict:
    root = directory.expanduser().resolve()
    required = ("map.pbstream", "map.yaml", "origin_reference.json",
                "navigation_config.json", "manifest.json")
    for name in required:
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise ValueError(f"incomplete deploy bundle: missing/empty {root / name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "fliftbot.deploy_catalog.v1":
        raise ValueError("unsupported deploy manifest schema")
    indexed = set()
    for item in manifest["files"]:
        name = item["path"]
        path = (root / name).resolve()
        if path.parent != root or name in indexed:
            raise ValueError(f"invalid or duplicate manifest path: {name}")
        indexed.add(name)
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"bundle size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"bundle checksum mismatch: {name}")
    for name in required[:-1]:
        if name not in indexed:
            raise ValueError(f"manifest does not cover {name}")
    image = yaml.safe_load((root / "map.yaml").read_text())["image"]
    if Path(image).name != image or image not in indexed:
        raise ValueError("occupancy image must be a local file covered by manifest")
    config = json.loads((root / "navigation_config.json").read_text(encoding="utf-8"))
    if config.get("schema") != "fliftbot.navigation_config.v1":
        raise ValueError("unsupported navigation config schema")
    if config.get("points_frame", "sensor") != "sensor":
        raise ValueError("navigation expects canonical points in the sensor frame")
    mount, odom = config["scan_to_base"], config["odometry"]
    values = [float(mount[key]) for key in ("x_m", "y_m", "yaw_deg")]
    if not math.isfinite(float(config["max_range_m"])) or config["max_range_m"] <= 0:
        raise ValueError("invalid max_range_m")
    values += [float(odom[key]) for key in
               ("speed_sign", "steering_sign", "steering_zero_deg", "wheelbase_m")]
    if not all(math.isfinite(v) for v in values) or odom["wheelbase_m"] <= 0:
        raise ValueError("invalid navigation calibration")
    if odom["speed_sign"] not in (-1, 1) or odom["steering_sign"] not in (-1, 1):
        raise ValueError("odometry signs must be -1 or +1")
    if odom["model"] not in ("drive_wheel", "bicycle", "legacy_sin"):
        raise ValueError("unsupported odometry model")
    reference = read_reference(root / "origin_reference.json")
    y_axis_sign = read_y_axis_sign(root / "origin_reference.json")
    # Load the actual raster too: a valid manifest alone does not validate YAML.
    from direct_udp_mc_localizer import load_map
    _, _, _, _, _ = load_map(root / "map.yaml")
    if (root / "pbstream_to_local_matrix.npy").is_file():
        matrix = np.load(root / "pbstream_to_local_matrix.npy")
        x, y, yaw = reference
        c, s = math.cos(yaw), math.sin(yaw)
        expected = np.array([[c, s, -c*x-s*y], [-s, c, s*x-c*y], [0, 0, 1]])
        expected[1] *= y_axis_sign
        if matrix.shape != (3, 3) or not np.allclose(matrix, expected, atol=1e-7):
            raise ValueError("origin_reference and pbstream_to_local_matrix disagree")
    # Web assets, when declared, must use the same origin and matrix chain.
    metadata_path = root / "map_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matrices = metadata.get("matrices", {})
        keys = {"local_xy_metres_to_raster_uv_pixels": "map_transform_matrix.npy",
                "pbstream_map_xy_metres_to_local_xy_metres": "pbstream_to_local_matrix.npy",
                "pbstream_map_xy_metres_to_raster_uv_pixels": "pbstream_to_raster_matrix.npy"}
        loaded = {}
        for key, filename in keys.items():
            if key not in matrices:
                continue
            path = root / filename
            if filename not in indexed or not path.is_file():
                raise ValueError(f"web matrix missing from bundle/manifest: {filename}")
            actual = np.load(path)
            if (actual.shape != (3, 3) or not np.isfinite(actual).all()
                    or not np.allclose(actual, matrices[key], atol=1e-7, rtol=0)):
                raise ValueError(f"web metadata matrix disagrees with {filename}")
            loaded[filename] = actual
        if len(loaded) == 3:
            local_to_pixel = loaded["map_transform_matrix.npy"]
            to_local = loaded["pbstream_to_local_matrix.npy"]
            if not np.allclose(local_to_pixel @ to_local,
                               loaded["pbstream_to_raster_matrix.npy"], atol=1e-7, rtol=0):
                raise ValueError("web and navigation matrix composition disagrees")
            origin_pixel = metadata.get("origin_pixel_xy_tver")
            if origin_pixel is not None and not np.allclose(
                    (local_to_pixel @ [0, 0, 1])[:2], origin_pixel, atol=1e-7, rtol=0):
                raise ValueError("web pixel origin disagrees with local origin")
        web_image = metadata.get("web_image")
        if web_image:
            if Path(web_image).name != web_image or web_image not in indexed:
                raise ValueError("web image must be local and covered by manifest")
            from PIL import Image
            with Image.open(root / web_image) as image:
                if list(image.size) != metadata.get("image_size_px"):
                    raise ValueError("web image size disagrees with metadata")
    return {"status": "accepted", "map_dir": str(root), "files_verified": len(indexed),
            "navigation_config": config, "origin_map_pose": reference}
