"""Two experiments the user asked for:
  (1) Tile the ELM champion's dense pattern across the WHOLE map.
      Champion packs dense R into one ~16-row band on a wire+road spine and
      hits 15,380. Here we tile that motif over all 100 rows: a connected
      wire grid (vertical wires + a horizontal bus) for power + horizontal
      roads for traffic, every zone touching both. Try several R/C/I mixes
      to probe whether residential DEMAND caps the scale-up.
  (2) Start from a WATER + FOREST map (engine terrain generator, no clearMap)
      instead of a bare cleared map, then build the same dense grid on land.
      Tests whether natural land value lifts the ~10-15k ceiling.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H

T = {"R": Tool.RESIDENTIAL, "C": Tool.COMMERCIAL, "I": Tool.INDUSTRIAL}

def run(env, e, ticks=400_000, sample=25000):
    peak = 0
    for t in range(ticks):
        e.simTick()
        if t % sample == 0 or t == ticks - 1:
            peak = max(peak, int(e.cityPop))
    s = env.stats
    return peak, s, e

def _land(e):
    """Boolean (H,W): True where buildable (not water 2..20)."""
    m = np.empty((WORLD_H, WORLD_W), np.uint16)
    for y in range(WORLD_H):
        for x in range(WORLD_W):
            m[y, x] = e.getTile(x, y) & 1023
    return ~((m >= 2) & (m <= 20))

def lay_grid(env, e, cell=4, mix=("R","R","R","C"), x0=1, y0=1, x1=WORLD_W-2, y1=WORLD_H-2,
             skip_water=False):
    """Connected wire grid (vertical wires every `cell` cols + horizontal bus
    at top) + horizontal roads every `cell` rows; dense zones at cell centers.
    With skip_water, only places on buildable land (routes around water)."""
    land = _land(e) if skip_water else None
    def ok(x, y):
        return (not skip_water) or land[y, x]
    busy = y0
    # if the bus row is mostly water, slide it down to a land-rich row
    if skip_water:
        landrows = land[y0:y1+1, x0:x1+1].sum(axis=1)
        busy = y0 + int(np.argmax(landrows))
    for x in range(x0, x1+1, cell):
        for y in range(y0, y1+1):
            if ok(x, y): env.place(Tool.WIRE, x, y)
    for x in range(x0, x1+1):
        if ok(x, busy): env.place(Tool.WIRE, x, busy)
    for y in range(y0, y1+1, cell):
        if y == busy: continue
        for x in range(x0, x1+1):
            if ok(x, y): env.place(Tool.ROAD, x, y)
    # power plants on land cells along the bus row
    placed_plants = 0
    for fx in (0.15, 0.3, 0.5, 0.7, 0.85):
        px = int(x0 + (x1-x0)*fx)
        if not skip_water or land[busy, px]:
            env.place(Tool.NUCLEARPOWER, px, busy); placed_plants += 1
    k = 0; n = 0
    for j in range((y1-y0)//cell):
        for i in range((x1-x0)//cell):
            zx, zy = x0 + i*cell + 2, y0 + j*cell + 2
            if zx >= x1-1 or zy >= y1-1:
                continue
            if skip_water and not land[zy, zx]:
                continue
            env.place(T[mix[k % len(mix)]], zx, zy); n += 1; k += 1
    return n

def fresh_cleared():
    env = MicropolisEnv(seed=42); e = env.engine
    e.clearMap(); e.setFunds(2_000_000_000); e.setAutoBudget(True)
    e.setEnableDisasters(False); e.setCityTax(0); e.setSpeed(3); e.setPasses(1)
    return env, e

def fresh_terrain():
    """Natural water+forest map: generateSomeCity makes rivers/lakes/trees;
    do NOT clearMap, so the land value source (terrain) is present."""
    env = MicropolisEnv(seed=42); e = env.engine
    e.generateSomeCity(42)          # rivers, lakes, forests
    e.setFunds(2_000_000_000); e.setAutoBudget(True)
    e.setEnableDisasters(False); e.setCityTax(0); e.setSpeed(3); e.setPasses(1)
    return env, e

def terrain_stats(e):
    # count water (2..20) and tree (21..39) tiles roughly
    m = np.empty((WORLD_H, WORLD_W), np.uint16)
    for y in range(WORLD_H):
        for x in range(WORLD_W):
            m[y, x] = e.getTile(x, y) & 1023
    water = int(((m >= 2) & (m <= 20)).sum())
    trees = int(((m >= 21) & (m <= 39)).sum())
    return water, trees

def main():
    print("=== EXP 1: tile the dense pattern across the whole map (cleared) ===")
    for mix in [("R",), ("R","R","R","C"), ("R","R","C","I"), ("R","C","I")]:
        env, e = fresh_cleared()
        nz = lay_grid(env, e, cell=4, mix=mix)
        peak, s, e = run(env, e)
        print(f"  mix={'/'.join(mix):9s} zones={nz:4d}  peak cityPop={peak:6d}  "
              f"R={s.res_pop} C={s.com_pop} I={s.ind_pop}  LV={e.landValueAverage}")

    print("\n=== EXP 2: start from a water+forest map (generated terrain) ===")
    env, e = fresh_terrain()
    w, t = terrain_stats(e)
    print(f"  terrain before build: water={w} trees={t} tiles")
    nz = lay_grid(env, e, cell=4, mix=("R","R","R","C"), skip_water=True)
    peak, s, e = run(env, e)
    print(f"  on-terrain RRRC (water-aware): zones={nz}  peak cityPop={peak}  "
          f"R={s.res_pop} C={s.com_pop} I={s.ind_pop}  LV={e.landValueAverage}")
    env2, e2 = fresh_cleared()
    nz2 = lay_grid(env2, e2, cell=4, mix=("R","R","R","C"))
    peak2, s2, e2 = run(env2, e2)
    print(f"  cleared   RRRC (baseline):    zones={nz2}  peak cityPop={peak2}  "
          f"R={s2.res_pop} C={s2.com_pop} I={s2.ind_pop}  LV={e2.landValueAverage}")


if __name__ == "__main__":
    main()
