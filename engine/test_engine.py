"""Smoke test for the headless Micropolis build.

Verifies: module imports, engine instantiates, tools place tiles,
simTick advances cityTime, state is readable.

Does NOT verify city growth dynamics — that depends on SimCity gameplay
mechanics (power coverage, road connectivity, R/C/I demand balance, ...)
and isn't an engine-integration concern.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build"))

import micropolisengine as me

# Tool IDs from src/micropolis.h enum EditingTool
TOOL_RESIDENTIAL = 0
TOOL_COMMERCIAL = 1
TOOL_INDUSTRIAL = 2
TOOL_WIRE = 6
TOOL_BULLDOZER = 7
TOOL_ROAD = 9
TOOL_COALPOWER = 13

# Fixed world dimensions (see src/map_type.h)
WORLD_W, WORLD_H = 120, 100


def make_engine(seed: int = 42, funds: int = 1_000_000) -> "me.Micropolis":
    eng = me.Micropolis()
    eng.resourceDir = os.path.join(HERE, "res") + os.sep
    # generateSomeCity runs the full new-city init path including doSimInit().
    eng.generateSomeCity(seed)
    eng.initGame()
    eng.clearMap()
    eng.setFunds(funds)
    eng.setGameLevel(2)
    eng.setAutoBulldoze(True)
    eng.setAutoBudget(True)
    eng.setEnableDisasters(False)
    eng.setSpeed(3)
    eng.setPasses(1)
    return eng


def main():
    eng = make_engine()
    assert eng.simSpeed == 3, eng.simSpeed
    assert not eng.simPaused
    print(f"engine ready: funds={eng.totalFunds} cityTime={eng.cityTime}")

    # Place a few tiles and verify state changed
    cx, cy = WORLD_W // 2, WORLD_H // 2
    eng.toolDown(TOOL_COALPOWER, cx, cy)
    eng.toolDown(TOOL_RESIDENTIAL, cx + 5, cy)
    eng.toolDown(TOOL_RESIDENTIAL, cx + 5, cy + 4)
    eng.toolDown(TOOL_ROAD, cx + 3, cy)
    plant_tile = eng.getTile(cx, cy) & 1023
    res_tile = eng.getTile(cx + 5, cy) & 1023
    road_tile = eng.getTile(cx + 3, cy) & 1023
    print(f"tile readback: plant={plant_tile} res={res_tile} road={road_tile}")
    assert plant_tile != 0, "plant placement didn't write tile"
    assert res_tile != 0,   "residential placement didn't write tile"

    # Tick the sim and confirm time advances
    t0 = eng.cityTime
    n_ticks = 20_000
    start = time.time()
    for _ in range(n_ticks):
        eng.simTick()
    elapsed = time.time() - start
    rate = n_ticks / elapsed
    print(f"ticks: {n_ticks} in {elapsed:.3f}s -> {rate:,.0f} ticks/sec")
    print(f"cityTime: {t0} -> {eng.cityTime}")
    assert eng.cityTime > t0, "sim is not advancing"
    print("OK")


if __name__ == "__main__":
    main()
