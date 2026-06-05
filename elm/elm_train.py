"""MAP-Elites with an LLM variation operator over code genomes.

Genome  = Python source defining act(obs, state) (a closed-loop tile-placement
          policy; see elm/sandbox.py for the contract).
Operator = Claude (mutation / crossover) — or a mock for offline testing.
Archive = elm/archive.py (same [0,1]^2 / 20x20 bins as qd_train.py).

Each iteration:
    1. sample parent(s) from the archive,
    2. ask the operator for a child policy,
    3. evaluate it on MicropolisEnv (sandboxed in a subprocess with a timeout),
    4. insert into the archive if it wins its behavior cell.

Run with the engine's Python:  /usr/bin/python3 -m elm.elm_train ...

Examples:
    # offline smoke (no API key, mock operator):
    /usr/bin/python3 -m elm.elm_train --operator mock --iters 30 --no-sandbox
    # real run:
    ANTHROPIC_API_KEY=... /usr/bin/python3 -m elm.elm_train \
        --operator claude --model claude-sonnet-4-6 --iters 200 \
        --measures res_ind --fitness dense --out results/elm_resind.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from elm.archive import MapElitesArchive
from elm.evaluate_code import eval_code, CodeResult, INVALID_FITNESS
from elm.operator import make_operator, OperatorError
from elm.seeds import SEEDS

MUTATE_DIRECTIVE = (
    "Produce a DIFFERENT city. Consider: changing the zone mix (add commercial "
    "and industrial, not just residential), the cluster size/shape, the road & "
    "wire layout, adding a second neighborhood, or reacting to obs.tile_map "
    "instead of a fixed plan. Aim to grow cityPop while landing in a different "
    "region of behavior space than the parent."
)
CROSSOVER_DIRECTIVE = (
    "Combine the strongest ideas from both parents into one coherent policy "
    "that grows a larger or more balanced city."
)


# ---- robust sandboxed evaluation (parallel subprocesses + averaging) ----
# The engine has a residual per-process non-determinism (uninitialized
# ASLR/pointer-order state the upstream determinism patches reduced but did not
# eliminate): the SAME code+seed scores ~19% noise across fresh processes. So a
# single eval is unreliable for archive selection. We average over `n_evals`
# independent rolls (distinct seeds, each its own process => independent hidden
# state) run in PARALLEL, so robustness costs little wall-clock. Each subprocess
# also guards against hangs/crashes via the wall-clock timeout.
def _eval_worker(q, idx, source, kwargs):
    try:
        q.put((idx, eval_code(source, **kwargs)))
    except Exception as e:  # noqa: BLE001
        q.put((idx, CodeResult(ok=False, fitness=INVALID_FITNESS,
                               measures=(0.0, 0.0),
                               error=f"worker: {type(e).__name__}: {e}")))


def robust_eval(source, timeout, n_evals, base_seed, **kwargs):
    """Run n_evals episodes (seeds base_seed..+n_evals-1) in parallel subprocesses
    and return (aggregated CodeResult with MEAN fitness/measures, [per-eval fits]).
    Aggregates only the episodes that succeeded; all-failed => ok=False."""
    import queue as _queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for i in range(n_evals):
        kw = dict(kwargs)
        kw["seed"] = base_seed + i
        kw["collect_render"] = (i == 0)   # one render is enough (for the operator)
        p = ctx.Process(target=_eval_worker, args=(q, i, source, kw))
        p.start()
        procs.append(p)

    results: dict = {}
    end = time.time() + timeout
    while len(results) < n_evals:
        remaining = end - time.time()
        if remaining <= 0:
            break
        try:
            idx, r = q.get(timeout=remaining)
            results[idx] = r
        except _queue.Empty:
            break
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join()

    ok = [results[i] for i in sorted(results) if results[i].ok]
    if not ok:
        err = next((results[i].error for i in sorted(results)),
                   f"timeout >{timeout}s / no results")
        return CodeResult(ok=False, fitness=INVALID_FITNESS, measures=(0.0, 0.0),
                          error=err), []
    import numpy as _np
    fits = [r.fitness for r in ok]
    render = next((results[i].render for i in sorted(results)
                   if results[i].render), None)
    agg = CodeResult(
        ok=True,
        fitness=float(_np.mean(fits)),
        measures=(float(_np.mean([r.measures[0] for r in ok])),
                  float(_np.mean([r.measures[1] for r in ok]))),
        city_pop=int(round(_np.mean([r.city_pop for r in ok]))),
        res_pop=int(round(_np.mean([r.res_pop for r in ok]))),
        com_pop=int(round(_np.mean([r.com_pop for r in ok]))),
        ind_pop=int(round(_np.mean([r.ind_pop for r in ok]))),
        n_actions_taken=ok[0].n_actions_taken,
        n_errors=sum(r.n_errors for r in ok),
        error=next((r.error for r in ok if r.error), None),
        render=render,
    )
    return agg, [round(f, 1) for f in fits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100, help="LLM variation steps")
    ap.add_argument("--operator", default="claude", choices=["claude", "mock"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seeds", default="plan",
                    help="comma list of seed genomes to inject: plan,random")
    ap.add_argument("--crossover-rate", type=float, default=0.25)
    ap.add_argument("--weighted-parents", action="store_true",
                    help="bias parent sampling toward higher fitness")
    # episode / scoring
    ap.add_argument("--n-actions", type=int, default=120)
    ap.add_argument("--ticks-per-action", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--episode-seed", type=int, default=42,
                    help="base seed; n_evals episodes use seed..seed+n_evals-1")
    ap.add_argument("--n-evals", type=int, default=5,
                    help="episodes averaged per genome (engine eval is ~19%% "
                         "noisy per process; averaging gives a robust fitness)")
    ap.add_argument("--fitness", default="dense",
                    choices=["pop", "dense", "varied", "growth"])
    ap.add_argument("--measures", default="res_ind",
                    choices=["road_ind", "res_ind", "density", "entropy_count"])
    ap.add_argument("--archive-dims", default="20,20")
    # sandbox / io
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="per-evaluation wall-clock budget (subprocess)")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="evaluate inline (faster, no timeout protection)")
    ap.add_argument("--out", default="results/elm_archive.npz")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="seed for operator/parent-selection RNG")
    ap.add_argument("--init-archive", default=None,
                    help="resume: load an existing ELM .npz archive and keep "
                         "evolving it (skips re-seeding). Lets a long run "
                         "survive container restarts by relaunching from its "
                         "last checkpoint.")
    args = ap.parse_args()

    dims = tuple(int(x) for x in args.archive_dims.split(","))
    rng = np.random.default_rng(args.rng_seed)
    if args.init_archive:
        archive = MapElitesArchive.load(args.init_archive)
        dims = tuple(int(x) for x in archive.dims)  # int (not np.int64) for json log
        print(f"resumed from {args.init_archive}: {archive.summary()}")
    else:
        archive = MapElitesArchive(dims=dims)

    eval_kwargs = dict(
        n_actions=args.n_actions, ticks_per_action=args.ticks_per_action,
        warmup_ticks=args.warmup, fitness_mode=args.fitness,
        measures_mode=args.measures,
    )

    def evaluate(source):
        """Return (aggregated CodeResult, [per-eval fitnesses])."""
        if args.no_sandbox:
            # inline averaging (no timeout protection) — for offline testing.
            rs = []
            for i in range(args.n_evals):
                rs.append(eval_code(source, seed=args.episode_seed + i,
                                    collect_render=(i == 0), **eval_kwargs))
            ok = [r for r in rs if r.ok]
            if not ok:
                return rs[0], []
            agg = CodeResult(
                ok=True, fitness=float(np.mean([r.fitness for r in ok])),
                measures=(float(np.mean([r.measures[0] for r in ok])),
                          float(np.mean([r.measures[1] for r in ok]))),
                city_pop=int(round(np.mean([r.city_pop for r in ok]))),
                res_pop=int(round(np.mean([r.res_pop for r in ok]))),
                com_pop=int(round(np.mean([r.com_pop for r in ok]))),
                ind_pop=int(round(np.mean([r.ind_pop for r in ok]))),
                n_errors=sum(r.n_errors for r in ok),
                error=next((r.error for r in ok if r.error), None),
                render=rs[0].render,
            )
            return agg, [round(r.fitness, 1) for r in ok]
        return robust_eval(source, args.timeout, args.n_evals,
                           args.episode_seed, **eval_kwargs)

    operator = make_operator(
        args.operator, rng=rng, model=args.model, temperature=args.temperature,
    ) if args.operator != "mock" else make_operator("mock", rng=rng)

    print(f"ELM: operator={args.operator} model={args.model if args.operator=='claude' else '-'}")
    print(f"  fitness={args.fitness} measures={args.measures} "
          f"archive={dims[0]}x{dims[1]} n_actions={args.n_actions} "
          f"ticks/action={args.ticks_per_action}")
    print(f"  base_seed={args.episode_seed} n_evals={args.n_evals} (avg over rolls); "
          f"sandbox={'off' if args.no_sandbox else f'on ({args.timeout}s)'}")

    # ---- full generation log ----------------------------------------------
    # Append-only JSONL capturing EVERY genome generated this run (seeds,
    # accepted, rejected, invalid, operator failures) with full source + outcome
    # — so the complete evolutionary trajectory can be analyzed afterwards, not
    # just the cell-winners that survive in the .npz archive.
    gen_log_path = args.out.rsplit(".", 1)[0] + "_generations.jsonl"
    gen_log = open(gen_log_path, "a")

    def log_gen(rec: dict):
        gen_log.write(json.dumps(rec)); gen_log.write("\n"); gen_log.flush()

    log_gen({"event": "run_start", "model": args.model, "operator": args.operator,
             "fitness": args.fitness, "measures": args.measures,
             "archive_dims": list(dims), "n_actions": args.n_actions,
             "ticks_per_action": args.ticks_per_action,
             "episode_seed": args.episode_seed, "iters": args.iters})
    print(f"  logging every generation -> {gen_log_path}")

    # ---- seed the archive (skipped when resuming: the loaded archive already
    #      carries an evolved front, and re-seeding would only waste evals) ----
    seed_names = [] if args.init_archive else args.seeds.split(",")
    for name in seed_names:
        name = name.strip()
        if name not in SEEDS:
            print(f"  ! unknown seed {name!r}, skipping")
            continue
        r, spread = evaluate(SEEDS[name])
        if not r.ok:
            print(f"  ! seed {name!r} failed: {r.error}")
            continue
        imp, cell = archive.add(SEEDS[name], r.fitness, r.measures,
                                city_pop=r.city_pop, origin="seed",
                                iteration=-1, render=r.render)
        log_gen({"event": "eval", "iteration": -1, "origin": "seed",
                 "name": name, "parents": [], "improved": bool(imp),
                 "cell": list(cell), "fitness": r.fitness, "measures": list(r.measures),
                 "city_pop": r.city_pop, "res_pop": r.res_pop, "com_pop": r.com_pop,
                 "ind_pop": r.ind_pop, "n_errors": r.n_errors, "error": r.error,
                 "eval_fits": spread, "source": SEEDS[name]})
        print(f"  seed {name:7s} fitness={r.fitness:8.1f} cityPop={r.city_pop:5d} "
              f"cell={cell} rolls={spread} {'+' if imp else 'x'}")
    if not archive.cells:
        print("no seeds inserted — aborting."); return
    archive.record_history(0)

    # ---- evolve ----
    t0 = time.time()
    n_improved = n_invalid = n_op_err = 0
    for it in range(1, args.iters + 1):
        do_cross = (len(archive.cells) >= 2 and rng.random() < args.crossover_rate)
        origin = "crossover" if do_cross else "mutate"
        parents = ()
        try:
            if do_cross:
                p1, p2 = archive.sample(rng, k=2, weighted=args.weighted_parents)
                parents = (p1.eid, p2.eid)
                child = operator.crossover(p1, p2, CROSSOVER_DIRECTIVE,
                                           renders=(p1.render, p2.render))
            else:
                (p,) = archive.sample(rng, k=1, weighted=args.weighted_parents)
                parents = (p.eid,)
                child = operator.mutate(p, MUTATE_DIRECTIVE, render=p.render)
        except OperatorError as e:
            n_op_err += 1
            log_gen({"event": "op_error", "iteration": it, "origin": origin,
                     "parents": list(parents), "error": str(e),
                     "source": getattr(e, "raw", None)})
            print(f"it {it:4d} [{('xover' if do_cross else 'mut')}] op-error: {e}")
            continue

        r, spread = evaluate(child)
        if not r.ok:
            n_invalid += 1
            log_gen({"event": "invalid", "iteration": it, "origin": origin,
                     "parents": list(parents), "error": r.error, "source": child})
            print(f"it {it:4d} [{origin[:5]}] invalid: {r.error}")
            continue

        imp, cell = archive.add(child, r.fitness, r.measures, city_pop=r.city_pop,
                                parents=parents, origin=origin, iteration=it,
                                render=r.render)
        log_gen({"event": "eval", "iteration": it, "origin": origin,
                 "parents": list(parents), "improved": bool(imp), "cell": list(cell),
                 "fitness": r.fitness, "measures": list(r.measures),
                 "city_pop": r.city_pop, "res_pop": r.res_pop, "com_pop": r.com_pop,
                 "ind_pop": r.ind_pop, "n_errors": r.n_errors, "error": r.error,
                 "eval_fits": spread, "source": child})
        n_improved += int(imp)
        b = archive.best
        flag = "+" if imp else " "
        print(f"it {it:4d} [{origin[:5]}] {flag} fit={r.fitness:8.1f} "
              f"pop={r.city_pop:5d} m=({r.measures[0]:.2f},{r.measures[1]:.2f}) "
              f"cell={cell} | filled={len(archive.cells)} "
              f"best_pop={b.city_pop} best_fit={b.fitness:.0f}")

        if it % args.save_every == 0:
            archive.record_history(it)
            archive.save(args.out)

    archive.record_history(args.iters)
    archive.save(args.out)
    log_gen({"event": "run_end", "improved": n_improved, "invalid": n_invalid,
             "op_errors": n_op_err, "filled": len(archive.cells)})
    gen_log.close()
    dt = time.time() - t0
    print("\n" + archive.summary())
    print(f"improved={n_improved} invalid={n_invalid} op_errors={n_op_err} "
          f"in {dt:.1f}s ({dt/max(1,args.iters):.2f}s/it)")
    print(f"saved archive -> {args.out}")
    print(f"saved full generation log -> {gen_log_path}")

    # Dump the champion's source for easy inspection / record.py replay.
    b = archive.best
    if b is not None:
        best_py = args.out.rsplit(".", 1)[0] + "_best.py"
        with open(best_py, "w") as f:
            f.write(f"# fitness={b.fitness:.1f} cityPop={b.city_pop} "
                    f"measures={b.measures} origin={b.origin}\n{b.source}\n")
        print(f"saved champion source -> {best_py}")


if __name__ == "__main__":
    main()
