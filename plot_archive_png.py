"""Render a QD archive (.npz from qd_train) as a PNG heatmap.

Usage:
    python3 plot_archive_png.py archive.npz heatmap.png
    python3 plot_archive_png.py archive.npz heatmap.png --title "tape@400"

Axes:
    X = measure 0 (road_frac, [0, 1])
    Y = measure 1 (ind_share, [0, 1])
    cell color = max fitness in that 1/20 x 1/20 cell (white = empty)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Plasma-ish 256-step palette built from 6 key stops.
_KEY_STOPS = [
    (13,   8,  135),
    (84,   2,  163),
    (158, 23,  173),
    (213, 64,  121),
    (248, 144,  53),
    (240, 249,  33),
]


def _build_palette() -> np.ndarray:
    n_seg = len(_KEY_STOPS) - 1
    pts_per_seg = 256 // n_seg
    out = np.zeros((256, 3), dtype=np.uint8)
    for s in range(n_seg):
        a = np.array(_KEY_STOPS[s], dtype=np.float32)
        b = np.array(_KEY_STOPS[s + 1], dtype=np.float32)
        for t in range(pts_per_seg):
            f = t / max(1, pts_per_seg - 1)
            out[s * pts_per_seg + t] = (a * (1 - f) + b * f).astype(np.uint8)
    out[-1] = _KEY_STOPS[-1]
    return out


_PAL = _build_palette()


def _color_for_value(v: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if vmax <= vmin:
        i = 128
    else:
        i = int(np.clip((v - vmin) / (vmax - vmin) * 255, 0, 255))
    return tuple(int(c) for c in _PAL[i])


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(
    archive_path: str,
    out_path: str,
    title: str = "",
    cell_size: int | None = None,
    grid_w: int | None = None,
    grid_h: int | None = None,
) -> None:
    d = np.load(archive_path, allow_pickle=True)
    objs = d["objectives"]
    measures = d["measures"]
    # Read grid_dims from metadata if available; fall back to 20x20.
    if grid_w is None or grid_h is None:
        if "grid_dims" in d.files:
            gd = d["grid_dims"]
            grid_w, grid_h = int(gd[0]), int(gd[1])
        else:
            grid_w, grid_h = 20, 20
    # Auto-pick cell size so the image stays roughly readable for any grid.
    if cell_size is None:
        cell_size = max(8, min(24, 480 // max(grid_w, grid_h)))

    # Bucket elites into a 20x20 grid (matches GridArchive in qd_train.py).
    grid = np.full((grid_h, grid_w), -np.inf, dtype=np.float32)
    for o, m in zip(objs, measures):
        ix = min(int(m[0] * grid_w), grid_w - 1)
        iy = min(int(m[1] * grid_h), grid_h - 1)
        if o > grid[iy, ix]:
            grid[iy, ix] = o

    # Heatmap colors
    filled = grid[grid > -np.inf]
    if len(filled) == 0:
        vmin = vmax = 0.0
    else:
        vmin, vmax = float(filled.min()), float(filled.max())

    # Layout: title + grid + colorbar + axes
    margin = 60
    title_h = 30 if title else 0
    grid_px_w = grid_w * cell_size
    grid_px_h = grid_h * cell_size
    cbar_w = 18
    cbar_pad = 50
    img_w = margin + grid_px_w + cbar_pad + cbar_w + margin
    img_h = margin + title_h + grid_px_h + margin

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_small = _load_font(11)
    font_mid = _load_font(13)
    font_big = _load_font(15)

    grid_x0 = margin
    grid_y0 = margin + title_h
    grid_x1 = grid_x0 + grid_px_w
    grid_y1 = grid_y0 + grid_px_h

    # Title (if any)
    if title:
        draw.text((grid_x0, margin // 2 - 4), title, fill=(0, 0, 0), font=font_big)

    # Draw cells (y axis flipped so high ind_share is at top)
    for iy in range(grid_h):
        for ix in range(grid_w):
            v = grid[iy, ix]
            x0 = grid_x0 + ix * cell_size
            y0 = grid_y1 - (iy + 1) * cell_size   # flipped
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            if v == -np.inf:
                fill = (245, 245, 245)
            else:
                fill = _color_for_value(v, vmin, vmax)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(220, 220, 220))

    # Axis labels
    draw.text(
        (grid_x0, grid_y1 + 8),
        "road_frac (0 → 1)", fill=(0, 0, 0), font=font_mid,
    )
    # Rotated y label: draw vertically character-by-character
    label = "ind_share (0 → 1)"
    cy = grid_y0 + grid_px_h // 2 - len(label) * 7 // 2
    for ch in label:
        draw.text((grid_x0 - 28, cy), ch, fill=(0, 0, 0), font=font_mid)
        cy += 12

    # Tick marks (just 0 and 1 corners)
    draw.text((grid_x0 - 4, grid_y1 + 2),  "0.0", fill=(80, 80, 80), font=font_small)
    draw.text((grid_x1 - 18, grid_y1 + 2), "1.0", fill=(80, 80, 80), font=font_small)
    draw.text((grid_x0 - 32, grid_y1 - 8), "0.0", fill=(80, 80, 80), font=font_small)
    draw.text((grid_x0 - 32, grid_y0 - 4), "1.0", fill=(80, 80, 80), font=font_small)

    # Colorbar
    cbar_x0 = grid_x1 + cbar_pad
    cbar_y0 = grid_y0
    cbar_y1 = grid_y1
    cbar_h = cbar_y1 - cbar_y0
    for i in range(cbar_h):
        # i=0 at top → highest value
        f = 1.0 - (i / max(1, cbar_h - 1))
        v = vmin + f * (vmax - vmin)
        c = _color_for_value(v, vmin, vmax)
        draw.line([(cbar_x0, cbar_y0 + i), (cbar_x0 + cbar_w, cbar_y0 + i)], fill=c)
    draw.rectangle([cbar_x0, cbar_y0, cbar_x0 + cbar_w, cbar_y1], outline=(120, 120, 120))
    draw.text((cbar_x0 + cbar_w + 4, cbar_y0 - 6),  f"{vmax:.0f}", fill=(0, 0, 0), font=font_small)
    draw.text((cbar_x0 + cbar_w + 4, cbar_y1 - 8),  f"{vmin:.0f}", fill=(0, 0, 0), font=font_small)
    draw.text((cbar_x0 + cbar_w + 4, (cbar_y0 + cbar_y1)//2 - 6), "obj", fill=(0, 0, 0), font=font_small)

    # Footer stats: elites filled
    n_filled = int((grid > -np.inf).sum())
    foot = f"{n_filled}/{grid_w*grid_h} cells filled    obj range [{vmin:.1f}, {vmax:.1f}]"
    draw.text((grid_x0, grid_y1 + 26), foot, fill=(60, 60, 60), font=font_small)

    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    render(args.archive, args.out, args.title)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
