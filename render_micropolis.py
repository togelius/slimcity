"""Tile renderer for slimcity using the real Micropolis GTK/web tileset.

Unlike record.py (which uses a synthetic per-category color table), this draws
each map tile with its actual 16x16 sprite from `engine/tiles.png` — the same
tileset the GTK frontend / Micropolis-web use — and animates via the engine's
own `animateTiles()` (the ANIMBIT tile-cycling that drives traffic, smokestack
puffs, the nuclear-plant swirl, fire, radar, etc.).

tiles.png is a 16-wide grid of 960 16x16 tiles: tile id N lives at
(row = N // 16, col = N % 16). Map tile ids come from getTile(x,y) & 1023.

CLI examples:
    # still PNG of a saved city, full 1920x1600
    python3 render_micropolis.py --city engine/cities/haight.cty --out docs/haight.png

    # animated GIF of a cropped, busy region (traffic + smoke move)
    python3 render_micropolis.py --city engine/cities/kowloon.cty \
        --gif docs/kowloon.gif --crop 40 30 80 70 --scale 3 --frames 48

    # render the best layout-genome elite from an archive
    python3 render_micropolis.py --layout-archive results/weekend_layout_resind.npz \
        --out docs/weekend_layout.png --gif docs/weekend_layout.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slimcity import MicropolisEnv, WORLD_W, WORLD_H

TILE = 16
DEFAULT_SHEET = os.path.join(HERE, "engine", "tiles.png")


class Tileset:
    """The Micropolis 16x16 tile sheet, sliced for vectorized blitting."""

    def __init__(self, path: str = DEFAULT_SHEET):
        sheet = np.asarray(Image.open(path).convert("RGB"))
        h, w, _ = sheet.shape
        self.cols = w // TILE
        self.rows = h // TILE
        self.n = self.cols * self.rows
        # [n, 16, 16, 3]
        self.tiles = (
            sheet.reshape(self.rows, TILE, self.cols, TILE, 3)
            .swapaxes(1, 2)
            .reshape(-1, TILE, TILE, 3)
        )

    def render(self, tile_map: np.ndarray) -> np.ndarray:
        """(H, W) tile-id array -> (H*16, W*16, 3) uint8 image, fully vectorized."""
        m = np.clip(tile_map, 0, self.n - 1).astype(np.intp)
        out = self.tiles[m]                       # (H, W, 16, 16, 3)
        h, w = m.shape
        return out.transpose(0, 2, 1, 3, 4).reshape(h * TILE, w * TILE, 3)


def _crop_scale(img: np.ndarray, crop, scale: int) -> Image.Image:
    if crop is not None:
        x0, y0, x1, y1 = crop
        img = img[y0 * TILE:y1 * TILE, x0 * TILE:x1 * TILE]
    pim = Image.fromarray(img)
    if scale != 1:
        pim = pim.resize((pim.width * scale, pim.height * scale), Image.NEAREST)
    return pim


def render_still(engine, tileset: Tileset, crop=None, scale: int = 1) -> Image.Image:
    m = _read_map(engine)
    return _crop_scale(tileset.render(m), crop, scale)


def _read_map(engine) -> np.ndarray:
    out = np.empty((WORLD_H, WORLD_W), dtype=np.uint16)
    for y in range(WORLD_H):
        for x in range(WORLD_W):
            out[y, x] = engine.getTile(x, y) & 1023
    return out


def render_gif(engine, tileset: Tileset, out_path: str, frames: int = 48,
               anim_steps_per_frame: int = 1, sim_ticks_per_frame: int = 0,
               crop=None, scale: int = 3, fps: int = 10) -> None:
    """Animate using the engine's own animateTiles(). With sim_ticks_per_frame>0
    the city also develops over the clip; with it 0 you get pure tile animation
    (traffic flowing, smokestacks puffing) on a frozen city."""
    imgs = []
    for _ in range(frames):
        for _ in range(sim_ticks_per_frame):
            engine.simTick()
        for _ in range(anim_steps_per_frame):
            engine.animateTiles()
        imgs.append(_crop_scale(tileset.render(_read_map(engine)), crop, scale))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)


# ---- subjects: a saved city, or a layout-genome archive's champion ----

def _engine_for_city(path: str):
    env = MicropolisEnv(seed=42, load_city=path)
    e = env.engine
    e.setSpeed(3); e.setPasses(1); e.setEnableDisasters(False)
    e.simTick()  # run census so dynamic tiles populate
    return env, e


def _engine_for_layout(archive: str):
    from policy import LayoutGenome
    d = np.load(archive, allow_pickle=True)
    theta = d["solutions"][int(np.argmax(d["objectives"]))].astype(np.float32)
    g = LayoutGenome(); g.set_params(theta)
    env = MicropolisEnv(seed=42); g.build(env)
    env.engine.setSpeed(3); env.engine.setPasses(1)
    env.tick(100000)  # stabilize so zones grow into their real sprites
    return env, env.engine


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--city", help="path to a .cty save file")
    src.add_argument("--layout-archive", help="layout-genome .npz; renders best elite")
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--out", default=None, help="still PNG output path")
    ap.add_argument("--gif", default=None, help="animated GIF output path")
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"), help="tile-coord crop box")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--sim-ticks-per-frame", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    tileset = Tileset(args.sheet)
    if args.city:
        env, e = _engine_for_city(args.city)
    else:
        env, e = _engine_for_layout(args.layout_archive)

    if args.out:
        render_still(e, tileset, crop=args.crop, scale=args.scale).save(args.out)
        print(f"wrote {args.out}")
    if args.gif:
        render_gif(e, tileset, args.gif, frames=args.frames,
                   sim_ticks_per_frame=args.sim_ticks_per_frame,
                   crop=args.crop, scale=max(args.scale, 1), fps=args.fps)
        print(f"wrote {args.gif}")
    if not args.out and not args.gif:
        print("nothing to do: pass --out and/or --gif")


if __name__ == "__main__":
    main()
