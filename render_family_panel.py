"""Render the best archive per approach family into a single comparison panel.

Handles both qd_train.py archive schema (objectives, solutions, grid_dims)
and ELM archive schema (fitness, sources, dims).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reuse the palette + helpers from plot_archive_png.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
sys.path.insert(0, "/Users/juliantogelius/Dropbox/nyckelpiga/slimcity")
from plot_archive_png import _build_palette, _color_for_value, _load_font

_PAL = _build_palette()


def load_archive(path):
    """Return dict {objectives, measures, grid_dims, measures_mode, label}."""
    d = np.load(path, allow_pickle=True)
    if "objectives" in d.files:
        out = {
            "objectives": d["objectives"],
            "measures": d["measures"],
            "grid_dims": tuple(int(x) for x in d["grid_dims"]) if "grid_dims" in d.files else (20, 20),
            "measures_mode": str(d["measures_mode"]) if "measures_mode" in d.files else "road_ind",
        }
    elif "fitness" in d.files:
        # ELM schema
        out = {
            "objectives": d["fitness"],
            "measures": d["measures"],
            "grid_dims": tuple(int(x) for x in d["dims"]) if "dims" in d.files else (20, 20),
            "measures_mode": "res_ind",   # ELM uses res_ind by default in our run
        }
    else:
        raise RuntimeError(f"unknown schema in {path}")
    return out


def render_cell_heatmap(d, title, cell_size=11):
    grid_w, grid_h = d["grid_dims"]
    grid = np.full((grid_h, grid_w), -np.inf, dtype=np.float32)
    for o, m in zip(d["objectives"], d["measures"]):
        ix = min(int(m[0] * grid_w), grid_w - 1)
        iy = min(int(m[1] * grid_h), grid_h - 1)
        if o > grid[iy, ix]:
            grid[iy, ix] = o
    filled = grid[grid > -np.inf]
    vmin, vmax = (float(filled.min()), float(filled.max())) if len(filled) else (0.0, 0.0)
    n_filled = int((grid > -np.inf).sum())

    margin = 16
    title_h = 20
    label_h = 14
    grid_px_w = grid_w * cell_size
    grid_px_h = grid_h * cell_size
    cbar_w = 12
    cbar_pad = 6
    w = margin + grid_px_w + cbar_pad + cbar_w + 40 + margin
    h = margin + title_h + grid_px_h + label_h + margin

    img = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    fbig = _load_font(13)
    fsm = _load_font(10)

    dr.text((margin, 4), title, fill=(0, 0, 0), font=fbig)

    gx0 = margin
    gy0 = margin + title_h
    gx1 = gx0 + grid_px_w
    gy1 = gy0 + grid_px_h

    for iy in range(grid_h):
        for ix in range(grid_w):
            v = grid[iy, ix]
            x0 = gx0 + ix * cell_size
            y0 = gy1 - (iy + 1) * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            if v == -np.inf:
                fill = (245, 245, 245)
            else:
                fill = _color_for_value(v, vmin, vmax)
            dr.rectangle([x0, y0, x1, y1], fill=fill, outline=(225, 225, 225))

    # colorbar
    cbar_x0 = gx1 + cbar_pad
    for i in range(grid_px_h):
        f = 1.0 - i / max(1, grid_px_h - 1)
        v = vmin + f * (vmax - vmin)
        c = _color_for_value(v, vmin, vmax)
        dr.line([(cbar_x0, gy0 + i), (cbar_x0 + cbar_w, gy0 + i)], fill=c)
    dr.rectangle([cbar_x0, gy0, cbar_x0 + cbar_w, gy1], outline=(120, 120, 120))
    dr.text((cbar_x0 + cbar_w + 2, gy0 - 4), f"{vmax:.0f}", fill=(0, 0, 0), font=fsm)
    dr.text((cbar_x0 + cbar_w + 2, gy1 - 8), f"{vmin:.0f}", fill=(0, 0, 0), font=fsm)

    # footer
    foot = f"{n_filled}/{grid_w*grid_h}  obj [{vmin:.0f}, {vmax:.0f}]  meas={d['measures_mode']}"
    dr.text((gx0, gy1 + 2), foot, fill=(60, 60, 60), font=fsm)
    return img


def main():
    RUNS = [
        ("results/exp_2026-06-03_layout_resind_50k.npz",         "layout @ 50k evals — cityPop 1,820"),
        ("results/elm_resind_sonnet_v2.npz",                     "ELM (Claude-sonnet) — cityPop 1,792"),
        ("results/overnight_tape_600_varied.npz",                "tape@600 — cityPop 1,120 (road_ind)"),
        ("results/longer_tape_600_resind.npz",                   "tape@600 — cityPop 1,120 (res_ind)"),
        ("results/wd_rich_hybrid_t200_500_growth_50g.npz",       "rich_hybrid t200@500 — cityPop 740"),
        ("results/wd_rich_randprefix_50_200_n5_growth_40g.npz",  "rich_randprefix — cityPop 360"),
        ("results/ec_rich_deepconv_growth_80g.npz",              "rich_deepconv growth — cityPop 0 (peak 119)"),
        ("results/ec_rich_deepconv_pop_80g.npz",                 "rich_deepconv pop — cityPop 0 (peak 0)"),
        ("results/overnight_randprefix_5x_long.npz",             "randprefix n=5 — cityPop 160"),
    ]
    imgs = [render_cell_heatmap(load_archive(p), title) for p, title in RUNS]

    iw, ih = imgs[0].size
    cols, rows = 3, 3
    pad = 6
    panel = Image.new("RGB",
                      (cols * iw + (cols + 1) * pad, rows * ih + (rows + 1) * pad),
                      (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        panel.paste(im, (pad + c * (iw + pad), pad + r * (ih + pad)))
    panel.save("results/family_comparison.png")
    print(f"wrote results/family_comparison.png  size={panel.size}")


if __name__ == "__main__":
    main()
