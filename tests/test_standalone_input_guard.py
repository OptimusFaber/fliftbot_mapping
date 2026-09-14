import json
import importlib.util
from pathlib import Path

import numpy as np



_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluation" / "check_standalone_inputs.py"
_SPEC = importlib.util.spec_from_file_location("check_standalone_inputs", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_standalone_inputs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_standalone_inputs)


def _recording(tmp_path: Path, *, gt_field: bool = False) -> Path:
    run = tmp_path / "nav_video_example"  # parent name is allowed
    horizon = run / "horizon"
    horizon.mkdir(parents=True)
    payload = {"xy": np.ones((12, 2), dtype=np.float32), "stamp": np.array(1.0)}
    if gt_field:
        payload["nav_x"] = np.zeros(1, dtype=np.float32)
    np.savez(horizon / "00001.npz", **payload)
    (run / "encoder.jsonl").write_text(
        json.dumps({"enc_unix": 1.0, "enc_v": 0.0, "enc_dgr": 0.0}) + "\n",
        encoding="utf-8",
    )
    return run


def test_guard_allows_nav_named_parent_and_real_inputs(tmp_path):
    assert check_standalone_inputs.audit(_recording(tmp_path)) == []


def test_guard_rejects_navigation_array(tmp_path):
    errors = check_standalone_inputs.audit(_recording(tmp_path, gt_field=True))
    assert any("GT/navigation arrays" in error for error in errors)


def test_guard_rejects_gt_encoder_fields(tmp_path):
    run = _recording(tmp_path)
    (run / "encoder.jsonl").write_text(
        json.dumps({"enc_unix": 1.0, "enc_v": 0.0, "enc_dgr": 0.0, "nav_x": 4.0}) + "\n",
        encoding="utf-8",
    )
    errors = check_standalone_inputs.audit(run)
    assert any("GT/navigation fields" in error for error in errors)
