"""Watch a city evolve in the terminal.

Three modes:
    --mode idle    : load preset, just tick — see how the preset evolves alone.
    --mode random  : load preset, random tools at random tiles between ticks.
    --mode policy  : load preset, drive with a random ConvPolicy.

Each frame prints the ASCII map plus a stats line. Use --delay to slow it
down enough to follow with the eye. Press Ctrl-C to stop.

Legend:
    R residential   C commercial   I industrial
    = roads/rails/wires   * power plant   ^ trees/parks
    ~ water    . empty land

Usage:
    python3 visualize.py
    python3 visualize.py --mode random --frames 40 --delay 0.3
    python3 visualize.py --mode idle --ticks-per-frame 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H
from random_smoke import sample_tool


def _clear_and_home():
    # ANSI: cursor home + clear-from-cursor-down
    sys.stdout.write("\033[H\033[J")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="",
                    help="preset .cty to load; default '' = empty map")
    ap.add_argument("--mode", choices=["idle", "random", "policy"], default="random")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--ticks-per-frame", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between frames")
    ap.add_argument("--downsample", type=int, default=2,
                    help="block size for ASCII render (2 -> 60x50 chars)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    city_path = os.path.join(here, args.city)
    print(f"loading {args.city}, warming up {args.warmup} ticks...")
    env = MicropolisEnv(seed=args.seed, load_city=city_path)
    env.tick(args.warmup)

    rng = np.random.default_rng(args.seed)

    policy = None
    if args.mode == "policy":
        from policy import ConvPolicy
        policy = ConvPolicy()
        # Small random weights — purely for visualization.
        policy.set_params(rng.standard_normal(ConvPolicy.param_count()).astype(np.float32) * 0.1)

    for f in range(args.frames):
        if args.mode == "random":
            tool = sample_tool(rng)
            x = int(rng.integers(0, WORLD_W))
            y = int(rng.integers(0, WORLD_H))
            env.place(tool, x, y)
            action_str = f"place {_tool_name(tool):<14} at ({x:>3},{y:>3})"
        elif args.mode == "policy":
            tool, x, y = policy.act(env.get_map())
            env.place(tool, x, y)
            action_str = f"place {_tool_name(tool):<14} at ({x:>3},{y:>3})"
        else:
            action_str = "(idle)"

        env.tick(args.ticks_per_frame)
        s = env.stats

        _clear_and_home()
        print(f"frame {f+1:>3}/{args.frames}   {action_str}")
        print(f"  city_time={s.city_time}  cityPop={s.city_pop:>6}  "
              f"R={s.res_pop:>4} C={s.com_pop:>3} I={s.ind_pop:>3}  "
              f"funds={s.funds:>8}  pollu={s.pollution_avg:>3}")
        print(env.render_ascii(color=not args.no_color, downsample=args.downsample))
        sys.stdout.flush()
        time.sleep(args.delay)

    print("\ndone.")


_TOOL_NAMES = {v: k for k, v in vars(Tool).items() if not k.startswith("_")}


def _tool_name(t: int) -> str:
    return _TOOL_NAMES.get(t, str(t))


if __name__ == "__main__":
    main()
