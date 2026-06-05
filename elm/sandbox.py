"""Compile and safely execute an evolved code genome.

The genome is a Python source string that must define a function:

    def act(obs, state):
        ...
        return None              # skip this step (place nothing)
        # or
        return (tool, x, y)      # place `tool` at tile (x, y)

`act` is called once per action step (see evaluate_code.eval_code), with
`ticks_per_action` ticks of simulation advancing between calls — the same
cadence as the float-vector closed-loop policies in policy.py. `state` is a
plain dict that persists across all steps of one episode (the policy's
memory); it starts empty each episode.

This module also defines PRIMITIVES_CARD — the contract handed to the LLM
operator and embedded in the seed docstrings — so the description of the API
lives in exactly one place.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import math
from dataclasses import dataclass

import numpy as np

from slimcity import WORLD_W, WORLD_H, Tool
from policy import RES_RANGE, COM_RANGE, IND_RANGE, ROAD_RANGE, PLANT_TILES

# Wire-only tile ids (conduct power; plain road does not). Mirrors the split
# used in evaluate.tile_descriptors.
WIRE_RANGE = (208, 222)
EMPTY_TILE = 0  # DIRT after clearMap()


# --------------------------------------------------------------------------
# Observation passed to act() each step.
# --------------------------------------------------------------------------
@dataclass
class Obs:
    """Snapshot of the game state at one step. Read-only as far as the policy
    is concerned (mutating tile_map has no effect on the engine)."""
    tile_map: np.ndarray   # (WORLD_H, WORLD_W)=(100,120) uint16 tile ids (low 10 bits)
    step: int              # 0-based index of this action step
    n_steps: int           # total action steps this episode
    city_pop: int
    res_pop: int
    com_pop: int
    ind_pop: int
    funds: int
    powered_zones: int     # engine.poweredZoneCount


# --------------------------------------------------------------------------
# Vectorized tile-category helpers exposed to the evolved code. These let a
# policy reason about the map without memorizing tile-id ranges. Each returns
# a boolean array the same shape as tile_map.
# --------------------------------------------------------------------------
def res_mask(tm):   return (tm >= RES_RANGE[0]) & (tm <= RES_RANGE[1])
def com_mask(tm):   return (tm >= COM_RANGE[0]) & (tm <= COM_RANGE[1])
def ind_mask(tm):   return (tm >= IND_RANGE[0]) & (tm <= IND_RANGE[1])
def road_mask(tm):  return (tm >= ROAD_RANGE[0]) & (tm <= ROAD_RANGE[1])
def wire_mask(tm):  return (tm >= WIRE_RANGE[0]) & (tm <= WIRE_RANGE[1])
def plant_mask(tm): return np.isin(tm, list(PLANT_TILES))
def empty_mask(tm): return tm == EMPTY_TILE


# --------------------------------------------------------------------------
# Restricted execution namespace.
#
# We deny imports / file IO / eval-exec so a stray (or pathological) genome
# can't touch the filesystem or network. This is a guard-rail, NOT a hard
# security boundary — the real isolation is the per-episode subprocess +
# timeout in elm_train.py. Keep the whitelist generous enough that ordinary
# numpy-flavored Python "just works" for the LLM.
# --------------------------------------------------------------------------
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "len", "list", "map", "max", "min", "pow", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "True", "False", "None", "isinstance", "print",
]
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES}


def make_namespace() -> dict:
    """Fresh globals dict for executing a genome."""
    ns = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "math": math,
        "Tool": Tool,
        "WORLD_W": WORLD_W,
        "WORLD_H": WORLD_H,
        "Obs": Obs,
        # tile-id ranges (low 10 bits)
        "RES_RANGE": RES_RANGE, "COM_RANGE": COM_RANGE, "IND_RANGE": IND_RANGE,
        "ROAD_RANGE": ROAD_RANGE, "WIRE_RANGE": WIRE_RANGE,
        "PLANT_TILES": PLANT_TILES, "EMPTY_TILE": EMPTY_TILE,
        # category mask helpers
        "res_mask": res_mask, "com_mask": com_mask, "ind_mask": ind_mask,
        "road_mask": road_mask, "wire_mask": wire_mask, "plant_mask": plant_mask,
        "empty_mask": empty_mask,
    }
    return ns


class CompileError(Exception):
    pass


def compile_policy(source: str):
    """Compile + exec `source`, returning its `act` callable.

    Raises CompileError if the source doesn't parse/exec or doesn't define a
    callable `act`. The returned closure shares the namespace, so any
    module-level helpers the genome defines are visible to act().
    """
    ns = make_namespace()
    try:
        code = compile(source, "<genome>", "exec")
        exec(code, ns)
    except Exception as e:  # noqa: BLE001 — surface any compile/exec failure
        raise CompileError(f"{type(e).__name__}: {e}") from e
    act = ns.get("act")
    if not callable(act):
        raise CompileError("genome must define a callable `act(obs, state)`")
    return act


# The optional open-loop init() bootstrap must stay small — a few lines that lay
# a starting base, NOT a full hand-coded city replayed open-loop.
INIT_MAX_LINES = 30


def _func_line_span(source: str, name: str) -> int:
    """Number of source lines spanned by the top-level function `name` (0 if absent)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            end = getattr(node, "end_lineno", None) or node.lineno
            return end - node.lineno + 1
    return 0


def compile_genome(source: str):
    """Compile + exec `source`, returning (act, init).

    `act(obs, state)` is REQUIRED — the per-step closed-loop policy.
    `init(obs, state)` is OPTIONAL — an open-loop bootstrap that returns a list
    of (tool, x, y) placements laid down BEFORE act() takes over. It must stay
    small (<= INIT_MAX_LINES lines) so the genome can't smuggle a full open-loop
    blueprint in through init.
    """
    ns = make_namespace()
    try:
        code = compile(source, "<genome>", "exec")
        exec(code, ns)
    except Exception as e:  # noqa: BLE001 — surface any compile/exec failure
        raise CompileError(f"{type(e).__name__}: {e}") from e
    act = ns.get("act")
    if not callable(act):
        raise CompileError("genome must define a callable `act(obs, state)`")
    init = ns.get("init")
    if init is not None and not callable(init):
        raise CompileError("`init` must be callable init(obs, state) if defined")
    if init is not None:
        span = _func_line_span(source, "init")
        if span > INIT_MAX_LINES:
            raise CompileError(
                f"init() is {span} lines; keep it <= {INIT_MAX_LINES} (~20 lines)")
    return act, init


def parse_action(action):
    """Validate/normalize a policy's return value.

    Returns (tool, x, y) with coords clamped into the map, or None to skip.
    Anything malformed -> None (treated as a skip, not a crash).
    """
    if action is None:
        return None
    try:
        tool, x, y = action          # must be a length-3 sequence
        tool = int(tool); x = int(x); y = int(y)
    except (TypeError, ValueError):
        return None
    if not (0 <= tool < 20):         # see slimcity.Tool (0..19)
        return None
    # Clamp coords rather than reject — friendlier to mutated code that drifts
    # a couple tiles out of bounds.
    x = max(0, min(WORLD_W - 1, x))
    y = max(0, min(WORLD_H - 1, y))
    return tool, x, y


# --------------------------------------------------------------------------
# The contract, in prose. Embedded in operator prompts and seed docstrings.
# --------------------------------------------------------------------------
PRIMITIVES_CARD = """\
You are evolving a Python policy that builds a city in a headless Micropolis
(SimCity 1989) engine. The map is a fixed 120 (wide) x 100 (tall) grid.

Define exactly one function:

    def act(obs, state):
        # Called once per step. Return None to place nothing this step,
        # or a (tool, x, y) tuple to place `tool` at tile column x, row y.
        ...

It is called `obs.n_steps` times. Between calls the simulation advances, so
zones you place earlier may grow before later calls. `state` is a dict that
persists across all steps of one episode (your memory); it starts empty.

You MAY also define an OPTIONAL open-loop bootstrap:

    def init(obs, state):
        # Runs ONCE, before act(). Return a list of (tool, x, y) placements that
        # lay a starting base (e.g. a power plant + a few wires/roads/zones).
        # They are applied one-per-step, then act() takes over for the rest.
        # Keep it SMALL — at most ~20 lines. It is open-loop scaffolding, not a
        # full pre-baked city.
        return [(Tool.COALPOWER, 58, 50), (Tool.WIRE, 58, 49), ...]

Use init() only to bootstrap a powered base so act() has something to react to;
the real, growing decisions should happen in the closed-loop act().

obs fields:
    obs.tile_map      numpy (100, 120) uint16 array of tile ids (rows=y, cols=x)
    obs.step          this step index (0-based)
    obs.n_steps       total steps
    obs.city_pop      current in-game population (the thing we maximize)
    obs.res_pop / obs.com_pop / obs.ind_pop / obs.funds
    obs.powered_zones number of zones currently on the power grid

Available names (no imports allowed; these are already in scope):
    np, math
    WORLD_W=120, WORLD_H=100
    Tool.RESIDENTIAL=0  COMMERCIAL=1  INDUSTRIAL=2  FIRESTATION=3
        POLICESTATION=4  WIRE=6  BULLDOZER=7  RAILROAD=8  ROAD=9
        STADIUM=10  PARK=11  SEAPORT=12  COALPOWER=13  NUCLEARPOWER=14  AIRPORT=15
    Tile-category masks over a tile_map (return boolean (100,120) arrays):
        res_mask, com_mask, ind_mask, road_mask, wire_mask, plant_mask, empty_mask
    Tile-id ranges: RES_RANGE, COM_RANGE, IND_RANGE, ROAD_RANGE, WIRE_RANGE, PLANT_TILES

Engine facts that matter for growth:
    - Zones (RESIDENTIAL/COMMERCIAL/INDUSTRIAL) are 3x3, placed by their CENTER.
    - A zone grows population only if it is POWERED and has ROAD access nearby.
    - WIRE conducts power; plain ROAD does NOT conduct power. You need at least
      one power plant (COALPOWER is cheap) connected via wires (or placed
      adjacent) to the zones.
    - Auto-bulldoze is ON: placing a tile clears what was under it, so don't
      lay a wire on top of your only power plant.
    - You have a limited number of steps, so you cannot tile the whole map —
      build a compact, well-powered, road-served cluster.

Return ONLY valid actions: tool in 0..15, x in 0..119, y in 0..99. Out-of-range
coords are clamped; a malformed return is treated as "skip". Keep the code
self-contained and deterministic given `state` (use np via a seed you stash in
state if you want randomness)."""
