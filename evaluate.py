"""Rollout a policy on MicropolisEnv and return (fitness, measures).

Fitness = delta in cityPop (in-game displayed population). We maximize.
Measures (QD descriptors, both in [0, 1]):
    0: road_frac  — road/rail/wire tiles / total built tiles
    1: ind_share  — industrial tiles / (R + C + I tiles)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import ConvPolicy, RES_RANGE, COM_RANGE, IND_RANGE, ROAD_RANGE, PLANT_TILES
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


def evaluate(
    theta: np.ndarray,
    seed: int = 0,
    n_actions: int = 50,
    ticks_per_action: int = 100,
    env: MicropolisEnv | None = None,
    city_path: str | None = DEFAULT_CITY,
    warmup_ticks: int = 500,
) -> EpisodeResult:
    if env is None:
        env = MicropolisEnv(seed=seed, load_city=city_path)
    else:
        env.reset(seed=seed)
    # Let the loaded city settle a bit before the agent acts.
    env.tick(warmup_ticks)
    baseline_pop = env.stats.city_pop

    policy = ConvPolicy()
    policy.set_params(theta)

    for _ in range(n_actions):
        tile_map = env.get_map()
        tool, wx, wy = policy.act(tile_map)
        env.place(tool, wx, wy)
        env.tick(ticks_per_action)

    s = env.stats
    fitness = float(s.city_pop - baseline_pop)
    measures = tile_descriptors(env.get_map())
    return EpisodeResult(fitness=fitness, measures=measures, stats_final=s)


if __name__ == "__main__":
    # quick smoke test
    import time
    np.random.seed(0)
    theta = np.random.randn(ConvPolicy.param_count()).astype(np.float32) * 0.1
    print(f"param count: {ConvPolicy.param_count()}")
    t0 = time.time()
    r = evaluate(theta, seed=42, n_actions=50, ticks_per_action=100)
    print(f"episode took {time.time()-t0:.2f}s")
    print(f"  fitness={r.fitness}  measures={r.measures}")
    print(f"  stats={r.stats_final}")
