"""Roll a code genome out on MicropolisEnv and score it.

Mirrors the semantics of evaluate._evaluate_once (fresh env per call for
determinism; same fitness modes and QD measures) but for a closed-loop
*code* policy with a no-op option, persistent per-episode state, and
per-step error tolerance.

Reuses the fitness-shaping and descriptor functions from evaluate.py so an
ELM archive is directly comparable to a CMA-ME archive.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from slimcity import MicropolisEnv
from evaluate import (
    tile_descriptors,
    _dense_bonus, _variety_bonus, _growth_bonus, _adjacency_bonus,
)
from elm.sandbox import Obs, compile_genome, parse_action, CompileError

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
    # --- closed-loopness instrumentation (filled when measures_mode=="cl_ind") ---
    cf_sensitivity: float = 0.0    # #2: fraction of steps whose action changes
                                   #     under an on-distribution counterfactual obs
    actions: tuple | None = None   # per-step parsed actions, for #3 cross-roll divergence


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


# --------------------------------------------------------------------------
# Closed-loopness descriptors (used when measures_mode == "cl_ind").
#
# #2 counterfactual-sensitivity: per step, re-run act() on a COPY of the state
#    with an ON-DISTRIBUTION donor observation (a real map+stats this same
#    rollout saw DONOR_LAG steps earlier, with the step index held fixed). If
#    the action changes, the policy genuinely used what it observed. Holding the
#    step fixed and copying the state isolate obs->action dependence and exclude
#    stochasticity (state-stashed RNG is copied, so both calls draw the same).
#
# #3 trajectory-divergence: computed across the n_evals rolls in robust_eval —
#    fraction of steps where the realized action sequences disagree between
#    different worlds. Captures state-mediated / accumulated reactivity (and
#    stochasticity) that #2's fixed-state probe can miss. Complementary to #2.
# --------------------------------------------------------------------------
DONOR_LAG = 10                 # how many steps back the counterfactual donor obs is
CL_IND_MODE = "cl_ind"


def trajectory_divergence(action_lists) -> float:
    """Mean pairwise fraction of steps where two rolls' action sequences differ."""
    lists = [a for a in action_lists if a]
    if len(lists) < 2:
        return 0.0
    tot, pairs = 0.0, 0
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            L = min(len(lists[i]), len(lists[j]))
            if L == 0:
                continue
            d = sum(lists[i][s] != lists[j][s] for s in range(L)) / L
            tot += d
            pairs += 1
    return tot / pairs if pairs else 0.0


def aggregate_measures(ok_results, measures_mode: str) -> tuple:
    """Combine per-roll results into the final QD descriptor.

    cl_ind -> 3-D (counterfactual_sensitivity, trajectory_divergence, ind_share),
    everything else -> 2-D mean of the per-roll tile_descriptors.
    """
    if measures_mode == CL_IND_MODE:
        cf = float(np.mean([r.cf_sensitivity for r in ok_results]))
        div = trajectory_divergence([r.actions for r in ok_results])
        ind = float(np.mean([r.measures[1] for r in ok_results]))  # res_ind -> [1]=ind
        return (cf, div, ind)
    return (float(np.mean([r.measures[0] for r in ok_results])),
            float(np.mean([r.measures[1] for r in ok_results])))


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
        act, init_fn = compile_genome(source)
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

    # ---- open-loop init phase ------------------------------------------------
    # Run the genome's optional init() once: it returns a list of (tool,x,y)
    # placements that lay a starting base BEFORE the closed-loop act() phase.
    # Capped to half the action budget so act() always has room to react (and so
    # a genome can't smuggle a full open-loop city through init).
    init_actions: list = []
    if init_fn is not None:
        s0 = env.stats
        init_obs = Obs(
            tile_map=env.get_map(), step=0, n_steps=n_actions,
            city_pop=s0.city_pop, res_pop=s0.res_pop, com_pop=s0.com_pop,
            ind_pop=s0.ind_pop, funds=s0.funds,
            powered_zones=env.engine.poweredZoneCount,
        )
        try:
            out = init_fn(init_obs, state)
        except TypeError:
            try:
                out = init_fn(state)          # tolerate an init(state) signature
            except Exception as e:  # noqa: BLE001
                out, n_err = None, n_err + 1
                first_err = first_err or f"init: {type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 — a bad init shouldn't kill the run
            out, n_err = None, n_err + 1
            first_err = first_err or f"init: {type(e).__name__}: {e}"
        if out:
            init_actions = list(out)[: n_actions // 2]
    init_len = len(init_actions)

    # Closed-loopness instrumentation (only when we're descriptor-ing on it).
    want_cl = (measures_mode == CL_IND_MODE)
    actions: list = []           # parsed action per step (for #3 cross-roll divergence)
    obs_hist: list = []          # past Obs, for the #2 counterfactual donor
    cf_probes = 0
    cf_changes = 0

    for step in range(n_actions):
        tile_map = env.get_map()
        s = env.stats
        obs = Obs(
            tile_map=tile_map, step=step, n_steps=n_actions,
            city_pop=s.city_pop, res_pop=s.res_pop, com_pop=s.com_pop,
            ind_pop=s.ind_pop, funds=s.funds,
            powered_zones=env.engine.poweredZoneCount,
        )

        # Open-loop init steps: replay the bootstrap, NO reactivity probe (these
        # are open-loop by design, so they must not count toward cf_sensitivity).
        if step < init_len:
            parsed = parse_action(init_actions[step])
            if parsed is not None:
                env.place(*parsed)
                n_taken += 1
            env.tick(ticks_per_action)
            continue

        # #2: counterfactual probe BEFORE the real act (needs the entering state).
        if want_cl and len(obs_hist) >= DONOR_LAG:
            d = obs_hist[-DONOR_LAG]
            donor = Obs(tile_map=d.tile_map, step=step, n_steps=n_actions,
                        city_pop=d.city_pop, res_pop=d.res_pop, com_pop=d.com_pop,
                        ind_pop=d.ind_pop, funds=d.funds,
                        powered_zones=d.powered_zones)
            try:
                a_cf = parse_action(act(donor, copy.deepcopy(state)))
            except Exception:  # noqa: BLE001
                a_cf = None
            cf_probes += 1
            # compared against the real action computed just below
        else:
            donor = None

        try:
            action = act(obs, state)
        except Exception as e:  # noqa: BLE001 — a bad step shouldn't kill the run
            n_err += 1
            if first_err is None:
                first_err = f"step {step}: {type(e).__name__}: {e}"
            action = None
        parsed = parse_action(action)

        if want_cl:
            actions.append(parsed)
            obs_hist.append(obs)
            if donor is not None and a_cf != parsed:
                cf_changes += 1

        if parsed is not None:
            env.place(*parsed)
            n_taken += 1
        env.tick(ticks_per_action)

    s = env.stats
    final_map = env.get_map()
    pop_delta = float(s.city_pop - baseline_pop)
    fitness = _shaped_fitness(pop_delta, final_map,
                              env.engine.poweredZoneCount, fitness_mode)
    # For cl_ind we still need ind_share from the tiles -> compute via res_ind.
    measures = tile_descriptors(final_map,
                                mode="res_ind" if want_cl else measures_mode)
    render = env.render_ascii(color=False) if collect_render else None
    cf_sensitivity = (cf_changes / cf_probes) if cf_probes else 0.0

    return CodeResult(
        ok=True, fitness=float(fitness), measures=measures,
        city_pop=s.city_pop, res_pop=s.res_pop, com_pop=s.com_pop,
        ind_pop=s.ind_pop, funds=s.funds,
        n_actions_taken=n_taken, n_errors=n_err, error=first_err, render=render,
        cf_sensitivity=cf_sensitivity, actions=(tuple(actions) if want_cl else None),
    )
