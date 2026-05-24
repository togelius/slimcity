"""CMA-ME training loop for MicropolisEnv using pyribs.

Fitness: delta in cityPop (in-game displayed population) — maximized.
Behavior characteristics (see evaluate.tile_descriptors):
    measure 0: road_frac  — roads as share of built tiles, [0, 1]
    measure 1: ind_share  — industry as share of zoned tiles, [0, 1]

Run a short smoke test:
    python3 qd_train.py --gens 5

Run a real (longer) experiment:
    python3 qd_train.py --gens 200 --emitters 5 --batch 20
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from evaluate import evaluate
from policy import ConvPolicy
from slimcity import MicropolisEnv


def build_scheduler(n_params: int, n_emitters: int, batch_size: int, sigma0: float,
                    es: str = "sep_cma_es"):
    archive = GridArchive(
        solution_dim=n_params,
        dims=[20, 20],                            # 20x20 = 400 archive cells
        ranges=[(0.0, 1.0), (0.0, 1.0)],          # (road_frac, ind_share)
        seed=0,
    )
    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=np.zeros(n_params, dtype=np.float32),
            sigma0=sigma0,
            ranker="2imp",        # CMA-ME improvement ranker
            es=es,                # "sep_cma_es" scales much better than full "cma_es"
            selection_rule="mu",
            restart_rule="basic",
            batch_size=batch_size,
            seed=i,
        )
        for i in range(n_emitters)
    ]
    return Scheduler(archive, emitters), archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10, help="number of generations")
    ap.add_argument("--emitters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=30, help="solutions per emitter per gen")
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--es", type=str, default="sep_cma_es",
                    help='ES algorithm: "sep_cma_es" (diagonal, scalable) or "cma_es" (full cov)')
    ap.add_argument("--n-actions", type=int, default=50)
    ap.add_argument("--ticks-per-action", type=int, default=100)
    ap.add_argument("--episode-seed", type=int, default=42,
                    help="fixed seed for env reset across all evals (deterministic fitness)")
    ap.add_argument("--save", type=str, default="archive.npz")
    args = ap.parse_args()

    n_params = ConvPolicy.param_count()
    print(f"policy parameters: {n_params}")

    scheduler, archive = build_scheduler(
        n_params=n_params,
        n_emitters=args.emitters,
        batch_size=args.batch,
        sigma0=args.sigma0,
        es=args.es,
    )

    # Reuse one env across evaluations - reset() is cheap.
    env = MicropolisEnv(seed=args.episode_seed)

    print(f"running {args.gens} generations, "
          f"{args.emitters} emitters x {args.batch} solutions = "
          f"{args.emitters * args.batch} evals/gen")

    t_start = time.time()
    for gen in range(args.gens):
        t0 = time.time()
        solutions = scheduler.ask()

        objectives = np.empty(len(solutions), dtype=np.float32)
        measures = np.empty((len(solutions), 2), dtype=np.float32)
        for i, sol in enumerate(solutions):
            r = evaluate(
                sol,
                seed=args.episode_seed,
                n_actions=args.n_actions,
                ticks_per_action=args.ticks_per_action,
                env=env,
            )
            objectives[i] = r.fitness
            measures[i] = r.measures

        scheduler.tell(objectives, measures)
        dt = time.time() - t0
        stats = archive.stats
        print(
            f"gen {gen+1:3d}/{args.gens}  "
            f"filled={stats.num_elites:4d}/{archive.cells}  "
            f"obj_max={stats.obj_max if stats.obj_max is not None else 0:.1f}  "
            f"obj_mean={stats.obj_mean if stats.obj_mean is not None else 0:.2f}  "
            f"qd_score={stats.qd_score:.1f}  "
            f"({dt:.1f}s, gen_max_obj={float(objectives.max()):.1f})"
        )

    print(f"\ntotal: {time.time()-t_start:.1f}s")
    # Save archive contents — ribs 0.8 API
    data = archive.data()  # dict of arrays: solution, objective, measures, ...
    np.savez(
        args.save,
        solutions=data["solution"],
        objectives=data["objective"],
        measures=data["measures"],
    )
    print(f"saved archive to {args.save} ({len(data['objective'])} elites)")


if __name__ == "__main__":
    main()
