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


def tile_descriptors(tile_map: np.ndarray,
                     mode: str = "road_ind") -> tuple[float, float]:
    """Compute a 2-D QD descriptor from a (H, W) tile-id map.

    Modes:
        road_ind   — (road_frac, ind_share) — the original.
            road_frac = roads / built; ind_share = industrial / zoned.
            Pure tape pinned all elites to the road_frac=0 column.
        res_ind    — (res_share, ind_share) within zoned tiles.
            Spreads policies along the R/C/I balance instead of by
            infrastructure. Should give tape something to explore beyond
            its current narrow niche.
        density    — (built_density, infra_density)
            built_density = (any built) / total tiles.
            infra_density = (road+wire+plant) / total tiles.
            Captures sprawl vs compactness independent of zone mix.
    """
    is_res = (tile_map >= RES_RANGE[0]) & (tile_map <= RES_RANGE[1])
    is_com = (tile_map >= COM_RANGE[0]) & (tile_map <= COM_RANGE[1])
    is_ind = (tile_map >= IND_RANGE[0]) & (tile_map <= IND_RANGE[1])
    is_road = (tile_map >= ROAD_RANGE[0]) & (tile_map <= ROAD_RANGE[1])
    is_plant = np.isin(tile_map, list(PLANT_TILES))

    built = is_res | is_com | is_ind | is_road | is_plant
    zoned = is_res | is_com | is_ind
    n_built = int(built.sum())
    n_road = int(is_road.sum())
    n_zone = int(zoned.sum())
    n_res = int(is_res.sum())
    n_ind = int(is_ind.sum())
    n_total = int(tile_map.size)
    n_infra = n_road + int(is_plant.sum())

    if mode == "road_ind":
        m0 = n_road / n_built if n_built > 0 else 0.0
        m1 = n_ind / n_zone if n_zone > 0 else 0.0
    elif mode == "res_ind":
        m0 = n_res / n_zone if n_zone > 0 else 0.0
        m1 = n_ind / n_zone if n_zone > 0 else 0.0
    elif mode == "density":
        m0 = n_built / n_total
        m1 = n_infra / n_total
    else:
        raise ValueError(f"unknown measures mode: {mode!r}")
    return float(m0), float(m1)


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


def _growth_bonus(tile_map: np.ndarray) -> float:
    """Count tiles that have transitioned past their ungrown state.

    Freshly placed 3×3 zones occupy tile IDs at the LOW end of each zone
    range (240-248 residential, 423-431 commercial, 612-620 industrial).
    When a zone grows the engine swaps in higher tile IDs. Counting tiles
    past the ungrown stamp gives intermediate credit BEFORE cityPop jumps
    up — useful gradient for "almost grew a zone" states the net half can
    optimize against.

    Scale: 0.5 per grown tile. A fully-grown 3×3 zone contributes ~4-9.
    """
    grown_res = ((tile_map > 248) & (tile_map <= 422)).sum()
    grown_com = ((tile_map > 431) & (tile_map <= 611)).sum()
    grown_ind = ((tile_map > 620) & (tile_map <= 692)).sum()
    return 0.5 * float(grown_res + grown_com + grown_ind)


def _adjacency_bonus(tile_map: np.ndarray) -> float:
    """Reward zones that are adjacent to power conductors and roads.

    These are necessary preconditions for growth in Micropolis. A zone
    surrounded by dirt cannot grow regardless of how much sim time passes,
    so rewarding spatial connection gives policies a path to climb that
    doesn't require accidentally satisfying every condition at once.

    Scale: 0.1 per (zone tile, neighbor) pair. A 3×3 zone fully bordered
    by road + plant on all sides gives ~3-9 per zone.
    """
    is_res = (tile_map >= RES_RANGE[0]) & (tile_map <= RES_RANGE[1])
    is_com = (tile_map >= COM_RANGE[0]) & (tile_map <= COM_RANGE[1])
    is_ind = (tile_map >= IND_RANGE[0]) & (tile_map <= IND_RANGE[1])
    zone = is_res | is_com | is_ind

    road  = (tile_map >= 64)  & (tile_map <= 95)
    wire  = (tile_map >= 208) & (tile_map <= 222)
    plant = np.isin(tile_map, list(PLANT_TILES))
    cond = road | wire | plant   # any power conductor

    def has_nbr(mask: np.ndarray) -> np.ndarray:
        """Returns a mask where each cell is True iff some 4-neighbor of it is in `mask`."""
        out = np.zeros_like(mask, dtype=bool)
        out[1:, :]  |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:]  |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    zone_with_cond_nbr = zone & has_nbr(cond)
    zone_with_road_nbr = zone & has_nbr(road)
    return 0.1 * float(zone_with_cond_nbr.sum() + zone_with_road_nbr.sum())


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
    measures_mode: str = "road_ind",
    n_evals: int = 1,
) -> EpisodeResult:
    """Run one (or n_evals averaged) episodes.

    When n_evals > 1, we run the same theta with seeds [seed, seed+1, ..., seed+n_evals-1]
    and average fitness + measures across runs. This is useful for stochastic
    policies (e.g. RandomPrefixPolicy) where any single episode is noisy.
    The returned stats_final is from the LAST run.
    """
    if n_evals > 1:
        results = []
        for i in range(n_evals):
            r = _evaluate_once(
                theta,
                seed=seed + i,
                n_actions=n_actions,
                ticks_per_action=ticks_per_action,
                env=None,  # always fresh env per eval
                city_path=city_path,
                warmup_ticks=warmup_ticks,
                policy_name=policy_name,
                policy_kwargs=policy_kwargs,
                fitness_mode=fitness_mode,
                measures_mode=measures_mode,
            )
            results.append(r)
        avg_fitness = float(np.mean([r.fitness for r in results]))
        avg_m0 = float(np.mean([r.measures[0] for r in results]))
        avg_m1 = float(np.mean([r.measures[1] for r in results]))
        return EpisodeResult(
            fitness=avg_fitness,
            measures=(avg_m0, avg_m1),
            stats_final=results[-1].stats_final,
        )
    return _evaluate_once(
        theta, seed=seed, n_actions=n_actions,
        ticks_per_action=ticks_per_action, env=env, city_path=city_path,
        warmup_ticks=warmup_ticks, policy_name=policy_name,
        policy_kwargs=policy_kwargs, fitness_mode=fitness_mode,
        measures_mode=measures_mode,
    )


def _evaluate_once(
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
    measures_mode: str = "road_ind",
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
    # Pass the seed to reset() — policies that need stochastic state (e.g.
    # RandomPrefixPolicy) use it; deterministic policies ignore it.
    try:
        policy.reset(seed=seed)
    except TypeError:
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
    elif fitness_mode == "growth":
        # varied + zone-growth + adjacency bonuses. Designed for closed-loop
        # policies that need gradient toward "your zone is connectable / your
        # zone just transitioned past ungrown".
        fitness = (
            pop_delta
            + _dense_bonus(final_map, env.engine.poweredZoneCount)
            + _variety_bonus(final_map)
            + _growth_bonus(final_map)
            + _adjacency_bonus(final_map)
        )
    else:
        fitness = pop_delta
    measures = tile_descriptors(final_map, mode=measures_mode)
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
