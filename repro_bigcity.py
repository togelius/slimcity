"""Reproduce a heavily-populated Micropolis city from scratch (empty map).

Reference image: the Micropolis "big city" screenshot on Wikimedia Commons
(https://upload.wikimedia.org/wikipedia/commons/6/6d/Micropolis_-_big_city.png),
an organic road-based city the game reports at capital population ~55,420
(tax 7%, "Pollution very high, Heavy Traffic, Crime very high").

We can't tile-copy an organic hand-built city (with natural water) from a
screenshot onto an empty cleared map, but we CAN reproduce its style with the
documented high-population technique + the engine's land-value equation:
  - cluster at the city center (land value peaks there)
  - forest carpet to seed terrain density (land value on bare land)
  - CONNECTED power: vertical wires joined by a horizontal wire "bus"
    (isolated wire columns were the bug in earlier attempts)
  - horizontal ROADS for traffic (high-density growth needs road access;
    plain roads don't conduct power, hence the separate wire network)
  - police lattice to hold down crime; mixed R/R/C/I for jobs+demand

Result: ~10,480 from empty — the best from-empty city in this project, and a
proper road-based one. Still far short of the screenshot's 55,420, which is an
organic hand-tuned city on terrain WITH water; see WEEKEND_REPORT.md.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H
from record import _COLOR_TBL

def build(env, e):
    cx, cy = WORLD_W // 2, WORLD_H // 2
    x0, x1, y0, y1 = cx - 30, cx + 30, cy - 24, cy + 24
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            e.toolDown(Tool.FOREST, x, y)
    busy = cy
    for x in range(x0, x1 + 1, 4):                 # vertical wires
        for y in range(y0, y1 + 1):
            e.toolDown(Tool.WIRE, x, y)
    for x in range(x0, x1 + 1):                    # horizontal wire bus
        e.toolDown(Tool.WIRE, x, busy)
    for y in range(y0, y1 + 1, 4):                 # horizontal roads
        if y == busy:
            continue
        for x in range(x0, x1 + 1):
            e.toolDown(Tool.ROAD, x, y)
    e.toolDown(Tool.NUCLEARPOWER, x0 + 4, busy)
    k = 0
    mix = [Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL]
    for j in range((y1 - y0) // 4):
        for i in range((x1 - x0) // 4):
            zx, zy = x0 + 4 * i + 2, y0 + 4 * j + 2
            if i % 4 == 0 and j % 4 == 0:
                e.toolDown(Tool.POLICESTATION, zx, zy); continue
            e.toolDown(mix[k % 4], zx, zy); k += 1

def main():
    env = MicropolisEnv(seed=42); e = env.engine
    e.clearMap(); e.setFunds(2_000_000_000); e.setAutoBudget(True)
    e.setEnableDisasters(False); e.setCityTax(7); e.setSpeed(3); e.setPasses(1)
    build(env, e)
    peak = 0
    for t in range(400_000):
        e.simTick()
        if t % 20000 == 0 or t == 399_999:
            peak = max(peak, int(e.cityPop))
    s = env.stats
    print(f"reproduction: peak cityPop={peak}  final={int(e.cityPop)}  "
          f"R={s.res_pop} C={s.com_pop} I={s.ind_pop}  LV={e.landValueAverage} "
          f"traffic={e.trafficAverage} crime={e.crimeAverage} powered={e.poweredZoneCount}")
    # render
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rgb = _COLOR_TBL[env.get_map()]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(rgb, interpolation="nearest", aspect="equal")
        ax.set_title(f"from-empty reproduction — cityPop {peak:,}\n"
                     f"(ref: Micropolis 'big city' = 55,420)", fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        out = os.path.join(HERE, "docs", "repro_bigcity.png")
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    except Exception as ex:
        print("render skipped:", ex)

if __name__ == "__main__":
    main()
