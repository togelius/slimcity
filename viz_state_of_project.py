"""Consolidated 'state of the project' visualization.

Produces docs/state_of_project.png with two rows:
  Row 1: best-cityPop leaderboard (horizontal bars, colored by family)
  Row 2: the two champion cities rendered side by side — layout (uniform
         mixing) vs ELM (structured neighborhoods) — both reaching ~1,800.

Run with the engine interpreter:
    /usr/bin/python3 viz_state_of_project.py
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from slimcity import MicropolisEnv, WORLD_W, WORLD_H
from policy import LayoutGenome
from record import _COLOR_TBL

# ELM rollout machinery
from elm.sandbox import compile_policy, parse_action, Obs


# ---------------- leaderboard data (from SUMMARY.md, replay-verified) -------
# (label, cityPop, family)
LEADER = [
    ("layout  res_ind 50k",        1820, "layout"),
    ("ELM  (Claude-sonnet, 148it)", 1792, "elm"),
    ("layout  res_ind 10k",        1680, "layout"),
    ("tape@600  varied",           1120, "tape"),
    ("tape@400  varied",            800, "tape"),
    ("rich_hybrid t200@500",        740, "hybrid"),
    ("tape@300  growth (ec)",       660, "tape"),
    ("tape@200  varied",            640, "tape"),
    ("rich_randprefix 50@200",      360, "randprefix"),
    ("randprefix n=5",              160, "randprefix"),
    ("any closed-loop net (scratch)", 0, "closedloop"),
]

FAMILY_COLOR = {
    "layout":     "#f59e0b",   # amber
    "elm":        "#8b5cf6",   # violet
    "tape":       "#2563eb",   # blue
    "hybrid":     "#0d9488",   # teal
    "randprefix": "#64748b",   # slate
    "closedloop": "#dc2626",   # red
}
FAMILY_LABEL = {
    "layout":     "Layout-CMA-ES (evolve the city)",
    "elm":        "ELM (LLM mutates Python code)",
    "tape":       "Open-loop ActionTape (CMA-ES)",
    "hybrid":     "Hybrid (tape + net)",
    "randprefix": "RandomPrefix + net",
    "closedloop": "Closed-loop nets from scratch",
}


# ---------------- render the layout champion --------------------------------
def render_layout_champion(archive="results/exp_2026-06-03_layout_resind_50k.npz"):
    d = np.load(archive, allow_pickle=True)
    sols, objs = d["solutions"], d["objectives"]
    theta = sols[int(np.argmax(objs))].astype(np.float32)
    g = LayoutGenome()
    g.set_params(theta)
    env = MicropolisEnv(seed=42)
    g.build(env)
    env.tick(100_000)
    s = env.stats
    return env.get_map(), s


# ---------------- render the ELM champion -----------------------------------
def render_elm_champion(archive="results/elm_resind_sonnet_v2.npz",
                        n_actions=120, ticks_per_action=100):
    d = np.load(archive, allow_pickle=True)
    cp, src = d["city_pop"], d["sources"]
    source = str(src[int(np.argmax(cp))])
    act = compile_policy(source)
    env = MicropolisEnv(seed=42)
    state: dict = {}
    for step in range(n_actions):
        tile_map = env.get_map()
        s = env.stats
        obs = Obs(tile_map=tile_map, step=step, n_steps=n_actions,
                  city_pop=s.city_pop, res_pop=s.res_pop, com_pop=s.com_pop,
                  ind_pop=s.ind_pop, funds=s.funds,
                  powered_zones=env.engine.poweredZoneCount)
        try:
            action = act(obs, state)
        except Exception:
            action = None
        parsed = parse_action(action)
        if parsed is not None:
            env.place(*parsed)
        env.tick(ticks_per_action)
    return env.get_map(), env.stats


def tilemap_to_rgb(tile_map):
    return _COLOR_TBL[tile_map]


def main():
    print("rendering layout champion ...")
    layout_map, layout_stats = render_layout_champion()
    print(f"  layout cityPop={layout_stats.city_pop} "
          f"R={layout_stats.res_pop} C={layout_stats.com_pop} I={layout_stats.ind_pop}")
    print("rendering ELM champion ...")
    elm_map, elm_stats = render_elm_champion()
    print(f"  ELM cityPop={elm_stats.city_pop} "
          f"R={elm_stats.res_pop} C={elm_stats.com_pop} I={elm_stats.ind_pop}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.12)

    # ---- leaderboard (spans top row) ----
    ax = fig.add_subplot(gs[0, :])
    labels = [x[0] for x in LEADER][::-1]
    vals   = [x[1] for x in LEADER][::-1]
    fams   = [x[2] for x in LEADER][::-1]
    colors = [FAMILY_COLOR[f] for f in fams]
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10, fontfamily="monospace")
    ax.set_xlabel("best replay-verified cityPop", fontsize=11)
    ax.set_title("Best city population by approach  (deterministic engine, replay-verified)",
                 fontsize=13, fontweight="bold", loc="left")
    for y, v in zip(ypos, vals):
        ax.text(v + 18, y, f"{v:,}", va="center", fontsize=9.5,
                fontweight="bold" if v >= 1700 else "normal")
    ax.set_xlim(0, 2050)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=FAMILY_COLOR[k], label=FAMILY_LABEL[k]) for k in FAMILY_LABEL]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    # ---- layout champion city ----
    axl = fig.add_subplot(gs[1, 0])
    axl.imshow(tilemap_to_rgb(layout_map), interpolation="nearest", aspect="equal")
    axl.set_title(f"Layout-CMA-ES champion — cityPop {layout_stats.city_pop:,}\n"
                  f"uniform random R/C/I mix, tax 0, wire grid",
                  fontsize=11, loc="left")
    axl.set_xticks([]); axl.set_yticks([])

    # ---- ELM champion city ----
    axe = fig.add_subplot(gs[1, 1])
    axe.imshow(tilemap_to_rgb(elm_map), interpolation="nearest", aspect="equal")
    axe.set_title(f"ELM champion — cityPop {elm_stats.city_pop:,}\n"
                  f"structured neighborhoods (wire spine + road + R/C/I bands)",
                  fontsize=11, loc="left")
    axe.set_xticks([]); axe.set_yticks([])

    fig.suptitle("slimcity — two routes to a ~1,800-pop Micropolis city",
                 fontsize=15, fontweight="bold", x=0.07, ha="left", y=0.98)
    out = os.path.join(HERE, "docs", "state_of_project.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
