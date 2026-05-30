"""Record a sim rollout as an animated GIF you can play in macOS Preview.

This is the practical alternative to a live Tk window: macOS Preview's
default GIF viewer gives you play/pause, scrubbing, and frame stepping
for free, and the rendering can't be broken by Tk quirks.

Usage:
    python3 record.py                                  # idle, ~6 sec
    python3 record.py --mode random --frames 200      # random play
    python3 record.py --mode policy --archive archive.npz
    python3 record.py --out city.gif --scale 6 --ticks-per-frame 30

Open the result:
    open city.gif
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H


# ---------- tile -> color lookup table (same as viewer.py) ----------

def _build_color_table() -> np.ndarray:
    tbl = np.zeros((1024, 3), dtype=np.uint8)
    tbl[:] = (210, 200, 165)  # empty land — sandy tan

    def shade(lo, hi, base, dark):
        n = hi - lo + 1
        for i in range(n):
            f = i / max(1, n - 1)
            tbl[lo + i] = (
                int(base[0] * (1 - f) + dark[0] * f),
                int(base[1] * (1 - f) + dark[1] * f),
                int(base[2] * (1 - f) + dark[2] * f),
            )

    for i in range(2, 20):   tbl[i] = (60, 160, 200)
    for i in range(21, 37):  tbl[i] = (40, 110, 50)
    for i in range(44, 48):  tbl[i] = (140, 100, 70)
    for i in range(48, 52):  tbl[i] = (80, 140, 180)
    for i in range(56, 64):  tbl[i] = (235, 90, 30)

    shade(64, 78,   (160, 160, 160), (90, 90, 90))
    shade(79, 95,   (120, 120, 120), (70, 70, 70))
    shade(96, 111,  (210, 200, 100), (170, 150, 60))
    shade(112, 175, (140, 140, 160), (90, 90, 110))
    shade(176, 206, (110, 90, 70),   (80, 60, 50))

    shade(240, 422, (180, 230, 180), (40, 140, 60))
    shade(423, 611, (180, 200, 240), (40, 80, 200))
    shade(612, 692, (230, 220, 140), (170, 140, 30))
    shade(693, 708, (90, 90, 140),   (60, 60, 110))
    shade(709, 744, (190, 190, 200), (140, 140, 160))
    shade(745, 760, (90, 70, 70),    (50, 30, 30))
    shade(761, 770, (230, 90, 60),   (180, 50, 30))
    shade(771, 778, (100, 130, 220), (60, 90, 180))
    shade(779, 810, (220, 130, 200), (170, 80, 150))
    shade(811, 826, (200, 90, 200),  (140, 40, 140))
    return tbl


_COLOR_TBL = _build_color_table()
_TOOL_NAMES = {v: k for k, v in vars(Tool).items() if not k.startswith("_")}


def sample_tool(rng: np.random.Generator) -> int:
    pool = [
        (Tool.RESIDENTIAL,   8),
        (Tool.COMMERCIAL,    4),
        (Tool.INDUSTRIAL,    4),
        (Tool.ROAD,         10),
        (Tool.WIRE,          4),
        (Tool.PARK,          2),
        (Tool.RAILROAD,      1),
        (Tool.POLICESTATION, 1),
        (Tool.FIRESTATION,   1),
    ]
    tools, weights = zip(*pool)
    p = np.array(weights, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(tools, p=p))


def render_frame(
    tile_map: np.ndarray, scale: int,
    stats, frame_idx: int, action_str: str,
    last_xy: tuple[int, int] | None,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame as a PIL Image: map on the left, stats on the right."""
    rgb = _COLOR_TBL[tile_map]
    if scale != 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    map_img = Image.fromarray(rgb, mode="RGB")

    map_w, map_h = WORLD_W * scale, WORLD_H * scale
    panel_w = 260
    total_w = map_w + panel_w
    total_h = map_h

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 240, 230))
    canvas.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Crosshair on last placement
    if last_xy is not None:
        x, y = last_xy
        cx = x * scale + scale // 2
        cy = y * scale + scale // 2
        r = max(scale * 2, 6)
        draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                       outline=(255, 40, 40), width=2)

    # Stats panel — dark text on light bg, monospace font
    panel_x = map_w + 12
    y = 12
    lines = [
        f"frame      {frame_idx}",
        f"city time  {stats.city_time}",
        f"cityPop    {stats.city_pop}",
        f"R/C/I      {stats.res_pop}/{stats.com_pop}/{stats.ind_pop}",
        f"funds      {stats.funds}",
        f"pollution  {stats.pollution_avg}",
        f"traffic    {stats.traffic_avg}",
        f"crime      {stats.crime_avg}",
        "",
        f"last act:",
        f"  {action_str}",
    ]
    for line in lines:
        draw.text((panel_x, y), line, fill=(30, 30, 30), font=font)
        y += 18

    # Legend
    y += 12
    draw.text((panel_x, y), "LEGEND", fill=(30, 30, 30), font=font)
    y += 22
    legend = [
        ((43, 140, 60),   "residential"),
        ((40, 80, 200),   "commercial"),
        ((170, 140, 30),  "industrial"),
        ((120, 120, 120), "road / rail"),
        ((210, 200, 100), "wire"),
        ((40, 110, 50),   "trees"),
        ((60, 160, 200),  "water"),
        ((140, 40, 140),  "power plant"),
        ((210, 200, 165), "empty land"),
    ]
    for color, name in legend:
        draw.rectangle([panel_x, y, panel_x + 14, y + 14], fill=color, outline=(60, 60, 60))
        draw.text((panel_x + 22, y), name, fill=(30, 30, 30), font=font)
        y += 20

    return canvas


def _load_font() -> ImageFont.ImageFont:
    """Pick a monospace font that exists on macOS, fall back to default."""
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, 13)
            except OSError:
                pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="",
                    help="preset .cty to load; default '' = empty map")
    ap.add_argument("--mode", choices=["idle", "random", "policy"], default="idle")
    ap.add_argument("--archive", default=None,
                    help="npz archive to load policy params from")
    ap.add_argument("--policy", default="conv",
                    help="policy representation for --mode policy: "
                         "conv | tape | ctxtape | mlp | deepconv")
    ap.add_argument("--policy-n-actions", type=int, default=None,
                    help="n_actions for tape/ctxtape policies (default = --frames)")
    ap.add_argument("--policy-hidden", type=int, default=32,
                    help="MLP hidden width (only for --policy mlp)")
    ap.add_argument("--policy-channels", type=str, default="16,32",
                    help="DeepConv channels comma-sep (only for --policy deepconv)")
    ap.add_argument("--scale", type=int, default=5, help="pixels per tile")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ticks-per-frame", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=0,
                    help="ticks before the policy starts acting (default 0 for empty maps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ms-per-frame", type=int, default=120,
                    help="GIF duration per frame in ms (lower = faster playback)")
    ap.add_argument("--out", default="city.gif")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    city_path = os.path.join(here, args.city) if args.city else None

    print(f"loading {args.city}, warming {args.warmup} ticks ...")
    env = MicropolisEnv(seed=args.seed, load_city=city_path)
    env.tick(args.warmup)

    rng = np.random.default_rng(args.seed)
    policy = None
    if args.mode == "policy":
        if not args.archive or not os.path.exists(args.archive):
            print(f"error: --mode policy needs --archive <path-to-npz>", file=sys.stderr)
            sys.exit(1)
        from policy import make_policy
        d = np.load(args.archive)
        if len(d["objectives"]) == 0:
            print("error: archive is empty", file=sys.stderr); sys.exit(1)
        idx = int(np.argmax(d["objectives"]))
        # Build per-policy kwargs
        policy_kwargs: dict = {}
        if args.policy in ("tape", "ctxtape"):
            n_act = args.policy_n_actions if args.policy_n_actions is not None else args.frames
            policy_kwargs = {"n_actions": n_act}
        elif args.policy == "mlp":
            policy_kwargs = {"hidden": args.policy_hidden}
        elif args.policy == "deepconv":
            policy_kwargs = {"channels": tuple(int(c) for c in args.policy_channels.split(","))}
        policy = make_policy(args.policy, **policy_kwargs)
        policy.set_params(d["solutions"][idx].astype(np.float32))
        policy.reset()
        print(f"  policy = {args.policy}{policy_kwargs}  elite: obj={float(d['objectives'][idx]):.1f}, "
              f"measures={d['measures'][idx]}")

    font = _load_font()

    print(f"recording {args.frames} frames ({args.mode}) ...")
    t0 = time.time()
    frames: list[Image.Image] = []

    # Initial frame (before any action)
    frames.append(render_frame(
        env.get_map(), args.scale, env.stats, 0, "(start)", None, font,
    ))

    for f in range(1, args.frames + 1):
        last_xy = None
        action_str = "(idle)"
        if args.mode == "random":
            tool = sample_tool(rng)
            x = int(rng.integers(0, WORLD_W))
            y = int(rng.integers(0, WORLD_H))
            env.place(tool, x, y)
            last_xy = (x, y)
            action_str = f"{_TOOL_NAMES.get(tool, '?')} @({x},{y})"
        elif args.mode == "policy":
            tool, x, y = policy.act(env.get_map())
            env.place(tool, x, y)
            last_xy = (x, y)
            action_str = f"{_TOOL_NAMES.get(tool, '?')} @({x},{y})"

        env.tick(args.ticks_per_frame)

        frames.append(render_frame(
            env.get_map(), args.scale, env.stats, f, action_str, last_xy, font,
        ))
        if f % 20 == 0:
            print(f"  frame {f}/{args.frames}  cityPop={env.stats.city_pop}")

    print(f"  capture done in {time.time()-t0:.1f}s; saving GIF ...")
    t0 = time.time()

    # Convert frames to a shared palette for smaller file size.
    # Pillow's "P" mode with optimize=True does this automatically.
    palette_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=256)
                      for f in frames]
    palette_frames[0].save(
        args.out,
        save_all=True,
        append_images=palette_frames[1:],
        duration=args.ms_per_frame,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"  wrote {args.out}  ({size_kb:.0f} KB, {len(frames)} frames, "
          f"{args.ms_per_frame}ms/frame, save took {time.time()-t0:.1f}s)")
    print(f"\nopen it with:  open {args.out}")


if __name__ == "__main__":
    main()
