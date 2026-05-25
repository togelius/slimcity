"""Rollout a policy on MicropolisEnv and return (fitness, measures).

Fitness modes:
    'pop'     — delta in cityPop (the in-game displayed population).
    'dense'   — pop + bonus for built tiles + bonus for powered zones; gives
                CMA-ME some signal even when no zone has actually grown yet.

Measures (QD descriptors, both in [0, 1]):
    0: road_frac  — road/rail/wire tiles / total built tiles
    1: ind_share  — industrial tiles / (R + C + I tiles)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import (
    ConvPolicy, make_policy,
    RES_RANGE, COM_RANGE, IND_RANGE, ROAD_RANGE, PLANT_TILES,
)
from slimcity import MicropolisEnv

import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Default = empty map (engine generateSomeCity + clearMap). To load a preset
# city, pass its path explicitly, e.g.
#   evaluate(theta, city_path=os.path.join(HERE, "engine/cities/radial.cty"))
DEFAULT_CITY: str | None = None


def tile_descriptors(tile_map: np.ndarray) -> tuple[float, float]:
    """Compute (road_frac, ind_share) from a (H, W) tile-id map.

    road_frac: roads/rails/wires as a share of all built (non-empty) tiles.
        0 = no infra (or empty map), 1 = all infra.
    ind_share: industrial as a share of zoned tiles (R+C+I).
        0 = no industry (or no zones), 1 = all industry.
    """
    is_res = (tile_map >= RES_RANGE[0]) & (tile_map <= RES_RANGE[1])
    is_com = (tile_map >= COM_RANGE[0]) & (tile_map <= COM_RANGE[1])
    is_ind = (tile_map >= IND_RANGE[0]) & (tile_map <= IND_RANGE[1])
    is_road = (tile_map >= ROAD_RANGE[0]) & (tile_map <= ROAD_RANGE[1])
    is_plant = np.isin(tile_map, list(PLANT_TILES))

    built = is_res | is_com | is_ind | is_road | is_plant
    n_built = int(built.sum())
    n_road = int(is_road.sum())
    n_zone = int((is_res | is_com | is_ind).sum())
    n_ind = int(is_ind.sum())

    road_frac = n_road / n_built if n_built > 0 else 0.0
    ind_share = n_ind / n_zone if n_zone > 0 else 0.0
    return float(road_frac), float(ind_share)


@dataclass
class EpisodeResult:
    fitness: float
    measures: tuple[float, float]
    stats_final: object  # slimcity.Stats


def _dense_bonus(tile_map: np.ndarray, powered_count: int) -> float:
    """A small shaped reward to give CMA-ME signal before any zone actually grows.

    The hard task (growing cityPop from empty) has a near-binary outcome —
    most policies get 0. This adds a bonus for things that are necessary
    prerequisites: building tiles at all, and getting zones onto the power
    grid. Scale is small relative to a working city (~1k+ cityPop).
    """
    is_built = tile_map != 0
    n_built = int(is_built.sum())
    return 0.01 * n_built + 1.0 * powered_count


def _variety_bonus(tile_map: np.ndarray) -> float:
    """Reward placing a variety of *categories* of tile.

    Six binary "have we ever placed one?" flags worth a few points each.
    Designed to break out of the initial degenerate state where a random
    policy spams a single tile type. Cheap, dense, monotone.

    Categories: residential / commercial / industrial / road / wire / plant.
    """
    is_res   = ((tile_map >= RES_RANGE[0])  & (tile_map <= RES_RANGE[1])).any()
    is_com   = ((tile_map >= COM_RANGE[0])  & (tile_map <= COM_RANGE[1])).any()
    is_ind   = ((tile_map >= IND_RANGE[0])  & (tile_map <= IND_RANGE[1])).any()
    # Road range covers roads, rails, wires; split coarsely:
    is_road  = ((tile_map >= 64)  & (tile_map <= 95)).any()    # road segments
    is_wire  = ((tile_map >= 208) & (tile_map <= 222)).any()   # wire segments
    is_plant = np.isin(tile_map, list(PLANT_TILES)).any()
    n_kinds = int(is_res) + int(is_com) + int(is_ind) + int(is_road) + int(is_wire) + int(is_plant)
    return 2.0 * n_kinds   # up to +12 with all six categories present


def evaluate(
    theta: np.ndarray,
    seed: int = 0,
    n_actions: int = 50,
    ticks_per_action: int = 100,
    env: MicropolisEnv | None = None,
    city_path: str | None = DEFAULT_CITY,
    warmup_ticks: int = 500,
    policy_name: str = "conv",
    policy_kwargs: dict | None = None,
    fitness_mode: str = "pop",
) -> EpisodeResult:
    # IMPORTANT: For determinism, always construct a fresh MicropolisEnv per
    # call. The C++ engine carries internal state (RNG depth into the LCG,
    # plus aux maps like landValueMap, pollutionMap, etc.) that env.reset()
    # does NOT fully restore — same theta+seed via env.reset() can give
    # different fitnesses across calls. A fresh env is ~0.5ms vs a ~450ms
    # episode, so the overhead is negligible.
    #
    # The `env` kwarg is accepted but ignored for the default deterministic
    # path; pass it only if you explicitly want the (non-deterministic) reuse
    # behavior — e.g. for interactive viewers.
    if env is None:
        env = MicropolisEnv(seed=seed, load_city=city_path)
    else:
        env.reset(seed=seed)
    if warmup_ticks:
        env.tick(warmup_ticks)
    baseline_pop = env.stats.city_pop

    policy = make_policy(policy_name, **(policy_kwargs or {}))
    policy.set_params(theta)
    policy.reset()

    for _ in range(n_actions):
        tile_map = env.get_map()
        tool, wx, wy = policy.act(tile_map)
        env.place(tool, wx, wy)
        env.tick(ticks_per_action)

    s = env.stats
    final_map = env.get_map()
    pop_delta = float(s.city_pop - baseline_pop)
    if fitness_mode == "dense":
        fitness = pop_delta + _dense_bonus(final_map, env.engine.poweredZoneCount)
    elif fitness_mode == "varied":
        # dense + variety bonus — helps non-tape policies escape the
        # zero-init degenerate state where every action lands on the same tile.
        fitness = (
            pop_delta
            + _dense_bonus(final_map, env.engine.poweredZoneCount)
            + _variety_bonus(final_map)
        )
    else:
        fitness = pop_delta
    measures = tile_descriptors(final_map)
    return EpisodeResult(fitness=fitness, measures=measures, stats_final=s)


if __name__ == "__main__":
    # quick smoke test
    import time
    np.random.seed(0)
    theta = np.random.randn(ConvPolicy.param_count()).astype(np.float32) * 0.1
    print(f"conv param count: {ConvPolicy.param_count()}")
    t0 = time.time()
    r = evaluate(theta, seed=42, n_actions=50, ticks_per_action=100, warmup_ticks=0)
    print(f"episode took {time.time()-t0:.2f}s")
    print(f"  fitness={r.fitness}  measures={r.measures}")
    print(f"  stats={r.stats_final}")
