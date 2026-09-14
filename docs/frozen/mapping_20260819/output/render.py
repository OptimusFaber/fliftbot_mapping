"""Render/output helpers backed by local mapping code."""

from __future__ import annotations

from mapping import live_map as _lm

save_global_map = _lm.save_global_map
save_local_png = _lm.save_local_png
save_local_frame = _lm.save_local_frame
