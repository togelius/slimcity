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
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from evaluate import evaluate
from policy import POLICY_REGISTRY, ConvPolicy, make_policy, policy_param_count
from slimcity import MicropolisEnv


# ---------------------------------------------------------------------------
# Multiprocessing workers
#
# Why processes instead of threads: the Micropolis engine is SWIG-wrapped C++
# and doesn't release the Python GIL during simTick / getTile loops, so real
# threads serialize anyway. Each worker is its own Python process with its
# own engine instance — fully independent, embarrassingly parallel.
#
# Workers persist for the lifetime of the Pool. Per-task overhead is just
# pickling theta (a 740-float32 array, ~3 KB) over a pipe. Initialization
# (spawn + import + engine construction) happens once per worker.
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker_init(episode_seed: int, n_actions: int,
                 ticks_per_action: int, warmup: int,
                 policy_name: str, policy_kwargs: dict,
                 fitness_mode: str, n_evals: int) -> None:
    """Run once per worker process. Stashes eval kwargs. The env is NOT
    cached here — see _worker_eval. We do warm the engine .so import path
    so it's loaded once."""
    # Import here so the engine .so is loaded in the worker, not the parent.
    import slimcity  # noqa: F401  -- import for side effect (load .so)
    _WORKER_STATE["kwargs"] = dict(
        seed=episode_seed,
        n_actions=n_actions,
        ticks_per_action=ticks_per_action,
        warmup_ticks=warmup,
        policy_name=policy_name,
        policy_kwargs=policy_kwargs,
        fitness_mode=fitness_mode,
        n_evals=n_evals,
    )


def _worker_eval(theta: np.ndarray) -> tuple[float, float, float]:
    """Evaluate one solution in a FRESH env. Determinism requires that the
    engine starts from a known C++ object state — env.reset() doesn't fully
    reset aux maps and RNG depth. Constructing a new MicropolisEnv is ~0.5ms,
    negligible vs the rollout cost."""
    from evaluate import evaluate as _evaluate
    # env=None tells evaluate() to construct a fresh one.
    r = _evaluate(theta, env=None, **_WORKER_STATE["kwargs"])
    return float(r.fitness), float(r.measures[0]), float(r.measures[1])


def build_scheduler(n_params: int, n_emitters: int, batch_size: int, sigma0: float,
                    es: str = "sep_cma_es",
                    grid_dims: tuple[int, int] = (20, 20)):
    archive = GridArchive(
        solution_dim=n_params,
        dims=list(grid_dims),                     # default 20x20 = 400 archive cells
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
    ap.add_argument("--warmup", type=int, default=0,
                    help="sim ticks to run before the policy acts. 0 makes sense "
                         "for empty-map starts; raise to ~500 if loading a preset.")
    ap.add_argument("--episode-seed", type=int, default=42,
                    help="fixed seed for env reset across all evals (deterministic fitness)")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of worker processes for parallel rollouts. "
                         "1 = sequential (no Pool). Recommended 4-8 on the M4 "
                         "(10 cores: 4 perf + 6 efficiency).")
    ap.add_argument("--policy", type=str, default="conv",
                    choices=sorted(POLICY_REGISTRY.keys()),
                    help="policy representation: conv | tape | mlp | deepconv")
    ap.add_argument("--policy-hidden", type=int, default=32,
                    help="MLP hidden width (only for --policy mlp)")
    ap.add_argument("--policy-channels", type=str, default="16,32",
                    help="DeepConv channels (comma sep). Used by --policy deepconv "
                         "and as the net half of --policy hybrid.")
    ap.add_argument("--policy-tape-len", type=int, default=50,
                    help="length of the tape prefix for --policy hybrid")
    ap.add_argument("--policy-random-len", type=int, default=50,
                    help="length of the random prefix for --policy randprefix")
    ap.add_argument("--fitness", type=str, default="pop",
                    choices=["pop", "dense", "varied", "growth"],
                    help="pop = cityPop delta; "
                         "dense = + built-tile and powered-zone bonus; "
                         "varied = dense + bonus for placing diverse tile categories; "
                         "growth = varied + grown-zone and zone-adjacent-to-infra bonuses")
    ap.add_argument("--n-evals", type=int, default=1,
                    help="evaluations per solution (averaged). >1 useful for "
                         "stochastic policies like randprefix; cost scales linearly.")
    ap.add_argument("--archive-dims", type=str, default="20,20",
                    help="GridArchive dims, comma-separated (e.g. 40,40 for 1600 cells)")
    ap.add_argument("--save", type=str, default="archive.npz")
    args = ap.parse_args()

    # Build policy kwargs based on the chosen policy
    policy_kwargs: dict = {}
    if args.policy == "tape":
        policy_kwargs = {"n_actions": args.n_actions}
    elif args.policy == "ctxtape":
        policy_kwargs = {"n_actions": args.n_actions}
    elif args.policy == "mlp":
        policy_kwargs = {"hidden": args.policy_hidden}
    elif args.policy == "deepconv":
        policy_kwargs = {"channels": tuple(int(c) for c in args.policy_channels.split(","))}
    elif args.policy == "hybrid":
        policy_kwargs = {
            "n_tape": args.policy_tape_len,
            "channels": tuple(int(c) for c in args.policy_channels.split(",")),
        }
        if args.policy_tape_len >= args.n_actions:
            print(f"warning: --policy-tape-len ({args.policy_tape_len}) >= --n-actions "
                  f"({args.n_actions}) — the net half will never run", file=sys.stderr)
    elif args.policy == "randprefix":
        policy_kwargs = {
            "n_random": args.policy_random_len,
            "channels": tuple(int(c) for c in args.policy_channels.split(",")),
        }
        if args.policy_random_len >= args.n_actions:
            print(f"warning: --policy-random-len ({args.policy_random_len}) >= --n-actions "
                  f"({args.n_actions}) — the net will never run", file=sys.stderr)
        if args.n_evals == 1:
            print("note: randprefix with --n-evals 1 only sees one random scaffold per "
                  "fitness eval — consider --n-evals 5 or more for noise smoothing.",
                  file=sys.stderr)

    if args.workers > 1:
        # Keep numpy/BLAS single-threaded inside workers so they don't all fight
        # over the same cores. Must be set before numpy is imported in workers.
        # spawn() forwards the parent env, so setting here propagates.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    n_params = policy_param_count(args.policy, **policy_kwargs)
    # Use policy-recommended sigma0 / ES if user didn't override (parser still
    # ran with defaults, so we only swap when defaults are present).
    proto = make_policy(args.policy, **policy_kwargs)
    if args.sigma0 == 0.5 and hasattr(proto, "SIGMA0"):
        args.sigma0 = proto.SIGMA0
    if args.es == "sep_cma_es" and hasattr(proto, "RECOMMENDED_ES"):
        args.es = proto.RECOMMENDED_ES
    print(f"policy = {args.policy}  params = {n_params}  sigma0 = {args.sigma0}  es = {args.es}")
    print(f"fitness = {args.fitness}")

    grid_dims = tuple(int(d) for d in args.archive_dims.split(","))
    assert len(grid_dims) == 2, f"expected 2 dims, got {grid_dims}"
    scheduler, archive = build_scheduler(
        n_params=n_params,
        n_emitters=args.emitters,
        batch_size=args.batch,
        sigma0=args.sigma0,
        es=args.es,
        grid_dims=grid_dims,
    )
    print(f"archive = {grid_dims[0]}x{grid_dims[1]} = {grid_dims[0]*grid_dims[1]} cells")

    # Set up either a sequential mode (workers=1) or a process Pool (workers>1).
    # Note: even in sequential mode we construct a fresh env per eval to keep
    # results deterministic — see evaluate.evaluate() for the rationale.
    pool = None
    if args.workers > 1:
        ctx = mp.get_context("spawn")  # spawn is the safe choice on macOS
        pool = ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.episode_seed, args.n_actions,
                      args.ticks_per_action, args.warmup,
                      args.policy, policy_kwargs, args.fitness,
                      args.n_evals),
        )
        print(f"running with {args.workers} worker processes")
    else:
        print("running sequentially (--workers 1)")

    print(f"running {args.gens} generations, "
          f"{args.emitters} emitters x {args.batch} solutions = "
          f"{args.emitters * args.batch} evals/gen")

    t_start = time.time()
    for gen in range(args.gens):
        t0 = time.time()
        solutions = scheduler.ask()

        objectives = np.empty(len(solutions), dtype=np.float32)
        measures = np.empty((len(solutions), 2), dtype=np.float32)

        if pool is not None:
            # pool.map preserves order: results[i] is for solutions[i].
            # Cast to float32 so pickled payloads are predictable and small.
            results = pool.map(
                _worker_eval,
                [s.astype(np.float32) for s in solutions],
            )
            for i, (f, m0, m1) in enumerate(results):
                objectives[i] = f
                measures[i, 0] = m0
                measures[i, 1] = m1
        else:
            for i, sol in enumerate(solutions):
                # env=None → evaluate() builds a fresh MicropolisEnv each
                # call. Required for deterministic fitness — see notes.
                r = evaluate(
                    sol,
                    seed=args.episode_seed,
                    n_actions=args.n_actions,
                    ticks_per_action=args.ticks_per_action,
                    warmup_ticks=args.warmup,
                    env=None,
                    policy_name=args.policy,
                    policy_kwargs=policy_kwargs,
                    fitness_mode=args.fitness,
                    n_evals=args.n_evals,
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
    # Save archive contents — ribs 0.8 API. Also store enough metadata that
    # any consumer (replay scripts, heatmap plotter, collaborator) can decode
    # the archive without needing the CLI args.
    data = archive.data()
    np.savez(
        args.save,
        solutions=data["solution"],
        objectives=data["objective"],
        measures=data["measures"],
        # Metadata
        grid_dims=np.array(grid_dims, dtype=np.int32),
        policy=np.array(args.policy),
        policy_kwargs=np.array(repr(policy_kwargs)),
        n_actions=np.int32(args.n_actions),
        ticks_per_action=np.int32(args.ticks_per_action),
        warmup=np.int32(args.warmup),
        fitness=np.array(args.fitness),
        n_evals=np.int32(args.n_evals),
        episode_seed=np.int32(args.episode_seed),
        gens=np.int32(args.gens),
        emitters=np.int32(args.emitters),
        batch=np.int32(args.batch),
        sigma0=np.float32(args.sigma0),
        es=np.array(args.es),
    )
    print(f"saved archive to {args.save} ({len(data['objective'])} elites)")

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
