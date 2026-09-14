#!/usr/bin/env python3
"""Interactive distance measurements on a ROS occupancy map.

Examples:
  python3 interactive_map_measure.py /path/to/map.yaml
  python3 interactive_map_measure.py /path/to/map.pgm --resolution 0.05

Ctrl+left-click creates point pairs.  The distance label is in centimetres
and is computed in the map's ROS world frame, not from the rotated display.
The GUI intentionally uses only Tkinter, Pillow and NumPy.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

DEFAULT_RESOLUTION = 0.05
MIN_ZOOM, MAX_ZOOM = 0.1, 12.0
ROTATION_STEP_DEG = 5.0
FINE_ROTATION_STEP_DEG = 1.0


def parse_map_yaml(path: Path) -> dict[str, Any]:
    """Read the small ROS map YAML subset without requiring PyYAML."""
    values: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key == "origin":
            value = value.strip("[]")
            try:
                values[key] = [float(x.strip()) for x in value.split(",")]
            except ValueError as exc:
                raise ValueError(f"Invalid origin in {path}: {value!r}") from exc
        elif key == "resolution":
            try:
                values[key] = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid resolution in {path}: {value!r}") from exc
        else:
            values[key] = value
    if "image" not in values:
        raise ValueError(f"{path} has no image entry")
    if "resolution" not in values or float(values["resolution"]) <= 0:
        raise ValueError(f"{path} must contain a positive resolution")
    origin = values.get("origin", [0.0, 0.0, 0.0])
    if len(origin) < 2:
        raise ValueError(f"{path} origin must contain at least x and y")
    values["origin"] = [float(origin[0]), float(origin[1]), float(origin[2]) if len(origin) > 2 else 0.0]
    return values


def resolve_map_input(path: Path, resolution: float = DEFAULT_RESOLUTION) -> tuple[Path, float, tuple[float, float]]:
    """Return image path, metres/pixel and ROS x/y origin."""
    path = path.expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        metadata = parse_map_yaml(path)
        image = Path(str(metadata["image"]))
        if not image.is_absolute():
            image = path.parent / image
        return image.resolve(), float(metadata["resolution"]), tuple(metadata["origin"][:2])
    if resolution <= 0:
        raise ValueError("--resolution must be positive")
    return path, float(resolution), (0.0, 0.0)


def pixel_to_world(pixel_xy: Any, image_height: int, resolution: float,
                   origin_xy: tuple[float, float]) -> np.ndarray:
    """Convert image coordinates (x=column, y=row; pixel centre) to ROS world."""
    p = np.asarray(pixel_xy, dtype=float)
    return np.stack((origin_xy[0] + (p[..., 0] + 0.5) * resolution,
                     origin_xy[1] + (image_height - p[..., 1] - 0.5) * resolution), axis=-1)


def world_distance_cm(a_pixel: Any, b_pixel: Any, image_height: int,
                      resolution: float, origin_xy: tuple[float, float]) -> float:
    """Return the physical distance between two image points in centimetres."""
    return float(np.linalg.norm(pixel_to_world(a_pixel, image_height, resolution, origin_xy)
                                - pixel_to_world(b_pixel, image_height, resolution, origin_xy)) * 100.0)


def rotate_display_point(pixel_xy: Any, original_size: tuple[int, int], angle_deg: float) -> np.ndarray:
    """Map an original image point to Pillow's expanded, visually rotated image."""
    w, h = original_size
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    p = np.asarray(pixel_xy, dtype=float)
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    q = p - center
    # y grows down on screen: this is positive visual counter-clockwise.
    rotated = np.stack((c * q[..., 0] + s * q[..., 1],
                        -s * q[..., 0] + c * q[..., 1]), axis=-1)
    out_w, out_h = rotated_image_size(original_size, angle_deg)
    return rotated + np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0])


@functools.lru_cache(maxsize=720)
def rotated_image_size(size: tuple[int, int], angle_deg: float) -> tuple[int, int]:
    """Match the dimensions produced by Pillow's rotate(..., expand=True)."""
    return Image.new("L", size).rotate(angle_deg, expand=True).size


def inverse_rotate_display_point(display_xy: Any, original_size: tuple[int, int],
                                 angle_deg: float) -> np.ndarray:
    """Inverse of rotate_display_point."""
    w, h = original_size
    out_w, out_h = rotated_image_size(original_size, angle_deg)
    q = np.asarray(display_xy, dtype=float) - np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0])
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    original = np.stack((c * q[..., 0] - s * q[..., 1],
                         s * q[..., 0] + c * q[..., 1]), axis=-1)
    return original + np.array([(w - 1) / 2.0, (h - 1) / 2.0])


@dataclass
class Segment:
    start: np.ndarray
    end: np.ndarray


class MeasurementGUI:
    def __init__(self, image_path: Path, resolution: float, origin: tuple[float, float]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("ROS occupancy map measurements")
        self.root.geometry("1400x950")
        self.root.minsize(800, 600)
        current_scaling = float(self.root.tk.call("tk", "scaling"))
        self.root.tk.call("tk", "scaling", max(current_scaling, 1.35))
        self.image_path, self.resolution, self.origin = image_path, resolution, origin
        self.image = Image.open(image_path).convert("RGBA")
        self.original_size = self.image.size
        self.segments: list[Segment] = []
        self.pending: np.ndarray | None = None
        self.angle = 0.0
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self._drag: tuple[float, float, np.ndarray] | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._render_key: tuple[Any, ...] | None = None
        self._rotated_image: Image.Image | None = None
        self._rotated_angle: float | None = None
        self._redraw_pending = False

        style = ttk.Style(self.root)
        style.configure("Measure.TButton", font=("TkDefaultFont", 12), padding=(12, 8))
        style.configure("Measure.TLabel", font=("TkDefaultFont", 11))

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=f"{image_path.name}  |  {resolution:g} m/px", style="Measure.TLabel").pack(side="left")
        ttk.Button(top, text="⟲  Q  −5°", style="Measure.TButton", command=lambda: self.rotate(-ROTATION_STEP_DEG)).pack(side="left", padx=(18, 4))
        ttk.Button(top, text="+5°  E  ⟳", style="Measure.TButton", command=lambda: self.rotate(ROTATION_STEP_DEG)).pack(side="left", padx=4)
        ttk.Button(top, text="Angle 0°", style="Measure.TButton", command=self.reset_angle).pack(side="left", padx=4)
        ttk.Button(top, text="Fit", style="Measure.TButton", command=self.fit).pack(side="left", padx=(14, 4))
        ttk.Button(top, text="Undo", style="Measure.TButton", command=self.undo).pack(side="left", padx=4)
        ttk.Button(top, text="Clear", style="Measure.TButton", command=self.clear).pack(side="left", padx=4)
        ttk.Button(top, text="Help", style="Measure.TButton", command=self.help).pack(side="right")
        self.status = ttk.Label(self.root, text="Ctrl+ЛКМ: первая точка", style="Measure.TLabel")
        self.status.pack(fill="x", padx=10)
        self.canvas = tk.Canvas(self.root, background="#30343b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        self.canvas.bind("<Configure>", lambda _e: self.request_redraw())
        self.canvas.bind("<Control-Button-1>", self.click)
        self.canvas.bind("<Button-1>", self.begin_pan)
        self.canvas.bind("<B1-Motion>", self.pan_motion)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", lambda e: self.wheel(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self.wheel(e, -1))
        self.root.bind("<Key-q>", lambda _e: self.rotate(-ROTATION_STEP_DEG))
        self.root.bind("<Key-e>", lambda _e: self.rotate(ROTATION_STEP_DEG))
        self.root.bind("<Shift-Key-Q>", lambda _e: self.rotate(-FINE_ROTATION_STEP_DEG))
        self.root.bind("<Shift-Key-E>", lambda _e: self.rotate(FINE_ROTATION_STEP_DEG))
        self.root.bind("<Key-r>", lambda _e: self.reset_angle())
        self.root.bind("<Key-u>", lambda _e: self.undo())
        self.root.bind("<Key-c>", lambda _e: self.clear())
        self.root.bind("<F1>", lambda _e: self.help())
        self.root.bind("<Escape>", self.cancel_pending)
        self.fit()

    def display_geometry(self) -> tuple[float, np.ndarray]:
        iw, ih = rotated_image_size(self.original_size, self.angle)
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        base = min((cw - 20) / iw, (ch - 20) / ih)
        scale = max(0.01, base * self.zoom)
        origin = np.array([cw / 2 - iw * scale / 2, ch / 2 - ih * scale / 2]) + self.pan
        return scale, origin

    def screen_to_pixel(self, x: float, y: float) -> np.ndarray | None:
        scale, origin = self.display_geometry()
        display = (np.array([x, y]) - origin) / scale
        rw, rh = rotated_image_size(self.original_size, self.angle)
        if not (0 <= display[0] < rw and 0 <= display[1] < rh):
            return None
        p = inverse_rotate_display_point(display, self.original_size, self.angle)
        if not (0 <= p[0] < self.original_size[0] and 0 <= p[1] < self.original_size[1]):
            return None
        return p

    def click(self, event: Any) -> str:
        p = self.screen_to_pixel(event.x, event.y)
        if p is None:
            return "break"
        if self.pending is None:
            self.pending = p
            self.status.configure(text="Первая точка выбрана — Ctrl+ЛКМ для второй")
        else:
            self.segments.append(Segment(self.pending, p))
            self.pending = None
            self.status.configure(text=f"Сегментов: {len(self.segments)} — Ctrl+ЛКМ для новой пары")
        self.request_redraw()
        return "break"

    def begin_pan(self, event: Any) -> None:
        if event.state & 0x0004:
            return
        self._drag = (event.x, event.y, self.pan.copy())

    def pan_motion(self, event: Any) -> None:
        if self._drag:
            x0, y0, p0 = self._drag
            self.pan = p0 + np.array([event.x - x0, event.y - y0])
            self.request_redraw()

    def wheel(self, event: Any, direction: int | None = None) -> None:
        direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        old_scale, old_origin = self.display_geometry()
        anchor = (np.array([event.x, event.y]) - old_origin) / old_scale
        self.zoom = float(np.clip(self.zoom * (1.15 if direction > 0 else 1 / 1.15), MIN_ZOOM, MAX_ZOOM))
        new_scale, _ = self.display_geometry()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        rw, rh = rotated_image_size(self.original_size, self.angle)
        centered = np.array([cw / 2 - rw * new_scale / 2, ch / 2 - rh * new_scale / 2])
        self.pan = np.array([event.x, event.y]) - anchor * new_scale - centered
        self.request_redraw()

    def rotate(self, degrees: float) -> None:
        self.angle = (self.angle + degrees) % 360
        self.status.configure(
            text=f"Угол: {self.angle:g}° — Ctrl+ЛКМ для измерения"
        )
        self.request_redraw()

    def reset_angle(self) -> None:
        self.angle = 0.0
        self.status.configure(text="Угол сброшен: 0° — Ctrl+ЛКМ для измерения")
        self.request_redraw()

    def fit(self) -> None:
        self.zoom, self.pan = 1.0, np.zeros(2)
        self.request_redraw()

    def undo(self) -> None:
        if self.pending is not None:
            self.pending = None
        elif self.segments:
            self.segments.pop()
        self.status.configure(text=f"Сегментов: {len(self.segments)}")
        self.request_redraw()

    def clear(self) -> None:
        self.pending, self.segments = None, []
        self.status.configure(text="Все измерения очищены — Ctrl+ЛКМ: первая точка")
        self.request_redraw()

    def cancel_pending(self, _event: Any = None) -> str:
        self.pending = None
        self.request_redraw()
        return "break"

    def help(self) -> None:
        from tkinter import messagebox
        messagebox.showinfo("Управление", "Ctrl+ЛКМ — поставить точки\n"
                            "ЛКМ + перетаскивание — перемещение\n"
                            "Колесо — масштабирование вокруг курсора\n"
                            "Q / E или кнопки — поворот на 5°\n"
                            "Shift+Q / Shift+E — точный поворот на 1°\n"
                            "R / Angle 0° — сбросить поворот\n"
                            "U / Undo — отменить точку или сегмент\n"
                            "C / Clear — очистить измерения\n"
                            "Esc — отменить незавершённую точку")

    def request_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            # Coalesce wheel/drag bursts to roughly one frame at 60 Hz.
            self.root.after(16, self.redraw)

    def redraw(self) -> None:
        self._redraw_pending = False
        scale, origin = self.display_geometry()
        if self._rotated_image is None or self._rotated_angle != self.angle:
            # Rotation is cached separately from zoom and pan.
            self._rotated_image = self.image.rotate(
                self.angle,
                expand=True,
                resample=Image.Resampling.NEAREST,
                fillcolor=(128, 128, 128, 255),
            )
            self._rotated_angle = self.angle
            self._render_key = None

        self.canvas.delete("all")
        rotated = self._rotated_image
        assert rotated is not None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        full_right = origin[0] + rotated.width * scale
        full_bottom = origin[1] + rotated.height * scale
        visible_left = max(0.0, origin[0])
        visible_top = max(0.0, origin[1])
        visible_right = min(float(cw), full_right)
        visible_bottom = min(float(ch), full_bottom)
        if visible_right > visible_left and visible_bottom > visible_top:
            # Crop in source pixels before resizing. At deep zoom this avoids
            # allocating a 10k×10k bitmap when only a window-sized tile is visible.
            crop_left = max(0, int(math.floor((visible_left - origin[0]) / scale)))
            crop_top = max(0, int(math.floor((visible_top - origin[1]) / scale)))
            crop_right = min(rotated.width, int(math.ceil((visible_right - origin[0]) / scale)))
            crop_bottom = min(rotated.height, int(math.ceil((visible_bottom - origin[1]) / scale)))
            crop_box = (crop_left, crop_top, crop_right, crop_bottom)
            shown_size = (
                max(1, int(round((crop_right - crop_left) * scale))),
                max(1, int(round((crop_bottom - crop_top) * scale))),
            )
            render_key = (round(self.angle, 6), round(scale, 7), crop_box, shown_size)
            if self._tk_image is None or self._render_key != render_key:
                tile = rotated.crop(crop_box)
                shown = tile.resize(shown_size, Image.Resampling.NEAREST)
                self._tk_image = ImageTk.PhotoImage(shown)
                self._render_key = render_key
            tile_origin = origin + np.asarray([crop_left, crop_top], dtype=float) * scale
            self.canvas.create_image(
                tile_origin[0], tile_origin[1], anchor="nw", image=self._tk_image
            )

        for index, segment in enumerate(self.segments, 1):
            a = origin + rotate_display_point(segment.start, self.original_size, self.angle) * scale
            b = origin + rotate_display_point(segment.end, self.original_size, self.angle) * scale
            self.canvas.create_line(*a, *b, fill="#ff4d4d", width=3)
            length = world_distance_cm(segment.start, segment.end, self.original_size[1],
                                       self.resolution, self.origin)
            mid = (a + b) / 2
            text_id = self.canvas.create_text(
                mid[0],
                mid[1] + 18,
                text=f"{length:.1f} cm",
                fill="#ffffff",
                font=("TkDefaultFont", 14, "bold"),
            )
            bbox = self.canvas.bbox(text_id)
            if bbox is not None:
                background_id = self.canvas.create_rectangle(
                    bbox[0] - 5,
                    bbox[1] - 3,
                    bbox[2] + 5,
                    bbox[3] + 3,
                    fill="#111827",
                    outline="#fbbf24",
                    width=2,
                )
                self.canvas.tag_lower(background_id, text_id)
            self.canvas.create_oval(a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7, fill="#ff4d4d", outline="white", width=2)
            self.canvas.create_oval(b[0] - 7, b[1] - 7, b[0] + 7, b[1] + 7, fill="#ff4d4d", outline="white", width=2)
            self.canvas.create_text(a[0] + 11, a[1] - 11, text=str(index), fill="white", anchor="sw", font=("TkDefaultFont", 12, "bold"))
        if self.pending is not None:
            p = origin + rotate_display_point(self.pending, self.original_size, self.angle) * scale
            self.canvas.create_oval(p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8, fill="#ffd166", outline="white", width=2)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", type=Path, help="map.yaml or direct PGM/PNG path")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION,
                        help=f"metres/pixel for direct image (default: {DEFAULT_RESOLUTION:g}); YAML overrides it")
    parser.add_argument("--no-show", action="store_true", help="load and validate the map, then exit without opening GUI")
    args = parser.parse_args(argv)
    try:
        image_path, resolution, origin = resolve_map_input(args.map, args.resolution)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        Image.open(image_path).verify()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.no_show:
        print(f"{image_path} | resolution={resolution:g} m/px | origin={origin}")
        return 0
    try:
        MeasurementGUI(image_path, resolution, origin).run()
    except ImportError as exc:
        print(f"GUI dependencies unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
