"""Random-action smoke test for MicropolisEnv.

Loads a preset city, warms it up, then places random tools at random tiles
while ticking the sim. Prints a population trajectory so we can eyeball
that the engine is doing something sensible.

Usage:
    python3 random_smoke.py
    python3 random_smoke.py --city engine/cities/radial.cty --actions 100
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H


# A "build something useful" subset of tools — bulldozer and query make no
# sense for random placement, and we don't want every other action to be a
# stadium or nuclear plant. Weights are rough; the point is to get a mix
# that has a chance of producing a working city.
TOOL_POOL = [
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


def sample_tool(rng: np.random.Generator) -> int:
    tools, weights = zip(*TOOL_POOL)
    p = np.array(weights, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(tools, p=p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="",
                    help="preset .cty to load; default '' = empty map")
    ap.add_argument("--actions", type=int, default=80)
    ap.add_argument("--ticks-per-action", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    city = os.path.join(here, args.city) if args.city else None

    print(f"loading {'preset ' + args.city if city else 'blank map'}")
    env = MicropolisEnv(seed=args.seed, load_city=city)

    rng = np.random.default_rng(args.seed)

    print(f"warming up {args.warmup} ticks...")
    env.tick(args.warmup)
    s0 = env.stats
    print(f"  after warmup: {_fmt(s0)}")

    print(f"\nrunning {args.actions} random actions x {args.ticks_per_action} ticks each\n")
    t_start = time.time()
    print(f"{'step':>4}  {'tool':>14}  {'(x,y)':>9}  {'res':>5}  {'com':>4}  "
          f"{'ind':>4}  {'cityPop':>7}  {'funds':>8}  {'pollu':>5}")
    print("-" * 78)
    for i in range(args.actions):
        tool = sample_tool(rng)
        x = int(rng.integers(0, WORLD_W))
        y = int(rng.integers(0, WORLD_H))
        env.place(tool, x, y)
        env.tick(args.ticks_per_action)
        s = env.stats
        if i < 10 or i % 10 == 9:
            print(f"{i+1:>4}  {_tool_name(tool):>14}  ({x:>3},{y:>3})  "
                  f"{s.res_pop:>5}  {s.com_pop:>4}  {s.ind_pop:>4}  "
                  f"{s.city_pop:>7}  {s.funds:>8}  {s.pollution_avg:>5}")

    s_end = env.stats
    dt = time.time() - t_start
    print("\nfinal:")
    print(f"  {_fmt(s_end)}")
    print(f"\ndelta vs post-warmup: "
          f"res {s_end.res_pop - s0.res_pop:+d}  "
          f"com {s_end.com_pop - s0.com_pop:+d}  "
          f"ind {s_end.ind_pop - s0.ind_pop:+d}  "
          f"city_pop {s_end.city_pop - s0.city_pop:+d}")
    print(f"\n{args.actions} actions in {dt:.2f}s "
          f"({args.actions / dt:.1f} actions/s, "
          f"{args.actions * args.ticks_per_action / dt:.0f} ticks/s)")


# ---------- helpers ----------

_TOOL_NAMES = {v: k for k, v in vars(Tool).items() if not k.startswith("_")}


def _tool_name(t: int) -> str:
    return _TOOL_NAMES.get(t, str(t))


def _fmt(s) -> str:
    return (f"city_time={s.city_time}  res={s.res_pop}  com={s.com_pop}  "
            f"ind={s.ind_pop}  cityPop={s.city_pop}  funds={s.funds}  "
            f"pollu={s.pollution_avg}")


if __name__ == "__main__":
    main()
