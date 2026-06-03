"""Roll a code genome out on MicropolisEnv and score it.

Mirrors the semantics of evaluate._evaluate_once (fresh env per call for
determinism; same fitness modes and QD measures) but for a closed-loop
*code* policy with a no-op option, persistent per-episode state, and
per-step error tolerance.

Reuses the fitness-shaping and descriptor functions from evaluate.py so an
ELM archive is directly comparable to a CMA-ME archive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slimcity import MicropolisEnv
from evaluate import (
    tile_descriptors,
    _dense_bonus, _variety_bonus, _growth_bonus, _adjacency_bonus,
)
from elm.sandbox import Obs, compile_policy, parse_action, CompileError

INVALID_FITNESS = -1e9  # genomes that fail to compile never win a cell


@dataclass
class CodeResult:
    ok: bool                       # False => compile failed; treat as invalid
    fitness: float
    measures: tuple                # (m0, m1), each in [0, 1]
    city_pop: int = 0
    res_pop: int = 0
    com_pop: int = 0
    ind_pop: int = 0
    funds: int = 0
    n_actions_taken: int = 0       # how many steps actually placed a tile
    n_errors: int = 0              # how many steps raised inside act()
    error: str | None = None       # compile error, or first runtime error repr
    render: str | None = None      # ascii city (if collect_render=True)


def _shaped_fitness(pop_delta: float, final_map: np.ndarray,
                    powered: int, mode: str) -> float:
    if mode == "dense":
        return pop_delta + _dense_bonus(final_map, powered)
    if mode == "varied":
        return (pop_delta + _dense_bonus(final_map, powered)
                + _variety_bonus(final_map))
    if mode == "growth":
        return (pop_delta + _dense_bonus(final_map, powered)
                + _variety_bonus(final_map) + _growth_bonus(final_map)
                + _adjacency_bonus(final_map))
    return pop_delta  # 'pop'


def eval_code(
    source: str,
    *,
    seed: int = 42,
    n_actions: int = 120,
    ticks_per_action: int = 100,
    warmup_ticks: int = 0,
    fitness_mode: str = "dense",
    measures_mode: str = "res_ind",
    city_path: str | None = None,
    collect_render: bool = False,
) -> CodeResult:
    """Compile `source`, run one episode, return a CodeResult.

    A fresh MicropolisEnv is constructed per call (the engine carries hidden
    RNG/aux-map state that reset() doesn't fully restore — see
    evaluate._evaluate_once). One episode is ~tens of ms.
    """
    try:
        act = compile_policy(source)
    except CompileError as e:
        return CodeResult(ok=False, fitness=INVALID_FITNESS, measures=(0.0, 0.0),
                          error=str(e))

    env = MicropolisEnv(seed=seed, load_city=city_path)
    if warmup_ticks:
        env.tick(warmup_ticks)
    baseline_pop = env.stats.city_pop

    state: dict = {}
    n_taken = 0
    n_err = 0
    first_err: str | None = None

    for step in range(n_actions):
        tile_map = env.get_map()
        s = env.stats
        obs = Obs(
            tile_map=tile_map, step=step, n_steps=n_actions,
            city_pop=s.city_pop, res_pop=s.res_pop, com_pop=s.com_pop,
            ind_pop=s.ind_pop, funds=s.funds,
            powered_zones=env.engine.poweredZoneCount,
        )
        try:
            action = act(obs, state)
        except Exception as e:  # noqa: BLE001 — a bad step shouldn't kill the run
            n_err += 1
            if first_err is None:
                first_err = f"step {step}: {type(e).__name__}: {e}"
            action = None
        parsed = parse_action(action)
        if parsed is not None:
            env.place(*parsed)
            n_taken += 1
        env.tick(ticks_per_action)

    s = env.stats
    final_map = env.get_map()
    pop_delta = float(s.city_pop - baseline_pop)
    fitness = _shaped_fitness(pop_delta, final_map,
                              env.engine.poweredZoneCount, fitness_mode)
    measures = tile_descriptors(final_map, mode=measures_mode)
    render = env.render_ascii(color=False) if collect_render else None

    return CodeResult(
        ok=True, fitness=float(fitness), measures=measures,
        city_pop=s.city_pop, res_pop=s.res_pop, com_pop=s.com_pop,
        ind_pop=s.ind_pop, funds=s.funds,
        n_actions_taken=n_taken, n_errors=n_err, error=first_err, render=render,
    )
