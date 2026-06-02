"""Thin Python wrapper around the headless Micropolis (SimCity 1) engine.

Designed for population-based / Quality-Diversity experiments where you need
many fast headless episodes. No Gymnasium dep — callers can wrap if needed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

# Locate the built C extension
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "engine", "build")
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import numpy as np
import micropolisengine as _me

# Tool IDs — keep in sync with src/micropolis.h enum EditingTool
class Tool:
    RESIDENTIAL = 0
    COMMERCIAL = 1
    INDUSTRIAL = 2
    FIRESTATION = 3
    POLICESTATION = 4
    QUERY = 5
    WIRE = 6
    BULLDOZER = 7
    RAILROAD = 8
    ROAD = 9
    STADIUM = 10
    PARK = 11
    SEAPORT = 12
    COALPOWER = 13
    NUCLEARPOWER = 14
    AIRPORT = 15
    NETWORK = 16
    WATER = 17
    LAND = 18
    FOREST = 19

WORLD_W = 120
WORLD_H = 100
TILE_MASK = 1023  # low 10 bits of getTile() are the tile id; upper bits are flags


@dataclass
class Stats:
    city_time: int
    res_pop: int
    com_pop: int
    ind_pop: int
    total_pop: int
    city_pop: int
    funds: int
    pollution_avg: int
    traffic_avg: int
    crime_avg: int


class MicropolisEnv:
    """Headless Micropolis sim with a Pythonic API.

    Notes:
        - The map is fixed 120 x 100 (compile-time).
        - generateSomeCity(seed) is used to run the engine's new-city init
          path (which clears initSimLoad and seeds power scan). We then
          clearMap() to start from blank land — terrain is gone after this,
          which is fine for placement-only experiments.
    """

    def __init__(
        self,
        seed: int = 0,
        funds: int = 1_000_000,
        disasters: bool = False,
        auto_budget: bool = True,
        auto_bulldoze: bool = True,
        speed: int = 3,
        passes: int = 1,
        game_level: int = 2,
        load_city: Optional[str] = None,
    ):
        """If `load_city` is a path to a `.cty` file, that city is loaded on
        each reset instead of starting from a blank generated map. Hand-built
        layouts from scratch rarely satisfy SimCity's growth conditions
        (power coverage, R/C/I demand balance), so loading a preset city is
        the practical way to get a sim that actually develops."""
        self._init_args = dict(
            funds=funds, disasters=disasters, auto_budget=auto_budget,
            auto_bulldoze=auto_bulldoze, speed=speed, passes=passes,
            game_level=game_level, load_city=load_city,
        )
        self._engine = _me.Micropolis()
        self._engine.resourceDir = os.path.join(_HERE, "engine", "res") + os.sep
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Stats:
        if seed is None:
            seed = int(np.random.randint(1, 2**15))
        a = self._init_args
        eng = self._engine
        if a["load_city"] is not None:
            ok = eng.loadCity(a["load_city"])
            if not ok:
                raise RuntimeError(f"loadCity failed: {a['load_city']}")
        else:
            eng.generateSomeCity(seed)
            eng.initGame()
            eng.clearMap()
        eng.setFunds(a["funds"])
        eng.setGameLevel(a["game_level"])
        eng.setAutoBulldoze(a["auto_bulldoze"])
        eng.setAutoBudget(a["auto_budget"])
        eng.setEnableDisasters(a["disasters"])
        eng.setSpeed(a["speed"])
        eng.setPasses(a["passes"])
        return self.stats

    def place(self, tool: int, x: int, y: int) -> int:
        """Place a tool at tile (x,y). Returns the engine's ToolResult code
        (0 = OK, 1 = need bulldoze, 2 = no money, etc.)."""
        return self._engine.toolDown(tool, x, y)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self._engine.simTick()

    def step(self, tool: int, x: int, y: int, sim_ticks: int = 1) -> Stats:
        """Place a tile then advance the sim by `sim_ticks`."""
        self.place(tool, x, y)
        self.tick(sim_ticks)
        return self.stats

    def get_map(self) -> np.ndarray:
        """Return a (WORLD_H, WORLD_W) uint16 array of tile IDs (low 10 bits)."""
        eng = self._engine
        out = np.empty((WORLD_H, WORLD_W), dtype=np.uint16)
        for y in range(WORLD_H):
            for x in range(WORLD_W):
                out[y, x] = eng.getTile(x, y) & TILE_MASK
        return out

    def get_map_raw(self) -> np.ndarray:
        """Return a (WORLD_H, WORLD_W) uint16 array of FULL tile values
        including status bits — PWRBIT (0x8000), CONDBIT (0x4000), ZONEBIT
        (0x0400), ANIMBIT, BURNBIT, BULLBIT.

        Useful for rich observations that want to know "is this tile
        currently powered?" (PWRBIT) without consulting the powerGridMap.
        """
        eng = self._engine
        out = np.empty((WORLD_H, WORLD_W), dtype=np.uint16)
        for y in range(WORLD_H):
            for x in range(WORLD_W):
                out[y, x] = eng.getTile(x, y)
        return out

    @property
    def stats(self) -> Stats:
        e = self._engine
        return Stats(
            city_time=e.cityTime,
            res_pop=e.resPop,
            com_pop=e.comPop,
            ind_pop=e.indPop,
            total_pop=e.totalPop,
            city_pop=int(e.cityPop),
            funds=int(e.totalFunds),
            pollution_avg=e.pollutionAverage,
            traffic_avg=e.trafficAverage,
            crime_avg=e.crimeAverage,
        )

    @property
    def engine(self) -> "_me.Micropolis":
        """Escape hatch: the raw SWIG-wrapped engine for advanced use."""
        return self._engine

    def render_ascii(self, color: bool = True, downsample: int = 2) -> str:
        """Return a coarse character map of the current city.

        Each output cell is a `downsample x downsample` block of the world
        (default 2 -> 60x50 chars, fits in a normal terminal). Categories:
            R/r residential   C/c commercial   I/i industrial
            =   road/rail/wire   *  power plant   ^  park/forest
            ~   water          .   land            ?  unknown
        Upper-case = denser category present in the block.
        """
        eng = self._engine
        H = WORLD_H // downsample
        W = WORLD_W // downsample
        out_lines = []
        for by in range(H):
            row = []
            for bx in range(W):
                ch = _BLANK
                # Pick the "most interesting" tile in the block — first match wins.
                cat = 0  # 0=land, will upgrade
                for dy in range(downsample):
                    for dx in range(downsample):
                        t = eng.getTile(bx * downsample + dx, by * downsample + dy) & TILE_MASK
                        c, c2 = _tile_category(t)
                        if c > cat:
                            cat = c
                            ch = c2
                row.append(_color(ch, cat) if color else ch)
            out_lines.append("".join(row))
        return "\n".join(out_lines)


# ---- ASCII render helpers ----
_BLANK = "."
# category rank — higher wins when picking a representative for a downsampled block
_CAT_WATER = 1
_CAT_PARK = 2
_CAT_ROAD = 3
_CAT_RES = 4
_CAT_COM = 5
_CAT_IND = 6
_CAT_PLANT = 7

# Tile-id ranges (low 10 bits). Kept loose; see policy.py for the same ranges.
_RES_LO, _RES_HI = 240, 422
_COM_LO, _COM_HI = 423, 611
_IND_LO, _IND_HI = 612, 692
_ROAD_LO, _ROAD_HI = 64, 206
_PLANT_TILES = {750, 816}
_WATER_LO, _WATER_HI = 2, 19   # river/water tiles in classic Micropolis
_TREE_LO, _TREE_HI = 21, 36


def _tile_category(tid: int) -> tuple[int, str]:
    if _RES_LO <= tid <= _RES_HI:
        return _CAT_RES, "R"
    if _COM_LO <= tid <= _COM_HI:
        return _CAT_COM, "C"
    if _IND_LO <= tid <= _IND_HI:
        return _CAT_IND, "I"
    if _ROAD_LO <= tid <= _ROAD_HI:
        return _CAT_ROAD, "="
    if tid in _PLANT_TILES:
        return _CAT_PLANT, "*"
    if _TREE_LO <= tid <= _TREE_HI:
        return _CAT_PARK, "^"
    if _WATER_LO <= tid <= _WATER_HI:
        return _CAT_WATER, "~"
    return 0, _BLANK


_ANSI = {
    _CAT_RES:   "\033[92m",   # green
    _CAT_COM:   "\033[94m",   # blue
    _CAT_IND:   "\033[93m",   # yellow
    _CAT_ROAD:  "\033[90m",   # bright black/gray
    _CAT_PLANT: "\033[95m",   # magenta
    _CAT_PARK:  "\033[32m",   # dark green
    _CAT_WATER: "\033[96m",   # cyan
}
_RESET = "\033[0m"


def _color(ch: str, cat: int) -> str:
    if cat in _ANSI:
        return f"{_ANSI[cat]}{ch}{_RESET}"
    return ch
