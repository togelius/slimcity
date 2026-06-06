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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from elm.archive import MapElitesArchive
from elm.evaluate_code import (
    eval_code, CodeResult, INVALID_FITNESS, aggregate_measures,
)
from elm.operator import make_operator, OperatorError
from elm.seeds import SEEDS

# Appended to the mutate-path directives: keep the operator pointed at the
# actual objective (population) without over-prescribing HOW — earlier, detailed
# "think big / more plants / use the whole map" language produced sprawled,
# fragmented cities that scored worse than a dense compact core.
AMBITION = (
    " Build a city that can support a LARGE POPULATION — maximizing the in-game "
    "population is the objective."
)
CLOSED_LOOP_REQUIRE = (
    " REQUIREMENT: act() must be genuinely CLOSED-LOOP — read obs each step and "
    "change what you do as the city grows; FULLY OPEN-LOOP replay policies are "
    "DISCARDED and score nothing. You MAY define a small open-loop init(obs, "
    "state) (<= ~20 lines) returning a list of (tool,x,y) to lay a starting base "
    "(plant + a few wires/roads); it runs first, then act() reacts and grows."
)

MUTATE_DIRECTIVE = (
    "Produce a DIFFERENT city. Consider: changing the zone mix (add commercial "
    "and industrial, not just residential), the cluster size/shape, the road & "
    "wire layout, adding a second neighborhood, or reacting to obs.tile_map "
    "instead of a fixed plan. Aim to grow cityPop while landing in a different "
    "region of behavior space than the parent."
    + AMBITION + CLOSED_LOOP_REQUIRE
)
CROSSOVER_DIRECTIVE = (
    "Combine the strongest ideas from both parents into one coherent policy "
    "that grows a larger or more balanced city."
)
# Used 50% of the time (CL_DEMAND_PROB) to explicitly push the operator toward
# genuinely reactive policies — the archive has cf_sensitivity / trajectory_
# divergence axes but nothing in the plain directives asks for closed-loop code,
# so the search otherwise stays 100% open-loop.
CLOSED_LOOP_DIRECTIVE = (
    "Write a genuinely CLOSED-LOOP policy. Each step, READ obs.tile_map (and "
    "obs.city_pop / obs.powered_zones) and decide the NEXT single placement "
    "from what you currently observe. Do NOT precompute a fixed list of "
    "placements and replay it; use `state` only as light memory (e.g. a small "
    "phase counter), never as a stored full plan. Concretely: scan the current "
    "map with the mask helpers (wire_mask, empty_mask, res_mask, road_mask, "
    "plant_mask), find where power and roads already reach, and place the next "
    "wire/road/zone adjacent to what already exists, adapting as the city "
    "grows. Still guarantee power (a coal plant + wires) and road access so "
    "zones actually grow. GOAL: the policy's actions should genuinely change "
    "when the observed map changes (high reactivity) AND it should still grow "
    "cityPop. This explores the high-reactivity region of the archive that "
    "open-loop replay policies cannot reach."
    + AMBITION +
    " You MAY add a small open-loop init(obs, state) (<= ~20 lines) that lays a "
    "starting base (plant + a few wires/roads); it runs first, then act() reacts."
)
# The crossover path also gets the ambition + closed-loop push.
CROSSOVER_DIRECTIVE = CROSSOVER_DIRECTIVE + AMBITION + CLOSED_LOOP_REQUIRE
CL_DEMAND_PROB = 0.5


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
        measures=aggregate_measures(ok, kwargs.get("measures_mode", "res_ind")),
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
                    choices=["road_ind", "res_ind", "density", "entropy_count",
                             "cl_ind"])
    ap.add_argument("--archive-dims", default="20,20",
                    help="comma dims; cl_ind is 3-D, e.g. 8,8,10 "
                         "(cf_sensitivity, traj_divergence, ind_share)")
    # sandbox / io
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent operator+eval pipelines (threads). Each does "
                         "its own LLM call + n_evals subprocesses; peak procs ~ "
                         "workers*n_evals. >1 keeps the CPU busy while LLM calls wait.")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="per-evaluation wall-clock budget (subprocess)")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="evaluate inline (faster, no timeout protection)")
    ap.add_argument("--out", default="results/elm_archive.npz")
    ap.add_argument("--resume", action="store_true",
                    help="resume from an existing --out archive + gen log "
                         "(skip seeding, continue iteration numbering)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="seed for operator/parent-selection RNG")
    ap.add_argument("--init-archive", default=None,
                    help="resume: load an existing ELM .npz archive and keep "
                         "evolving it (skips re-seeding). Lets a long run "
                         "survive container restarts by relaunching from its "
                         "last checkpoint.")
    ap.add_argument("--min-reactivity", type=float, default=-1.0,
                    help="DISCARD fully open-loop genomes: requires measures=cl_ind. "
                         "A genome is killed if max(cf_sensitivity, trajectory_"
                         "divergence) <= this. Default -1.0 disables it; pass 0.0 "
                         "to kill genomes that show no reactivity at all.")
    args = ap.parse_args()

    dims = tuple(int(x) for x in args.archive_dims.split(","))
    rng = np.random.default_rng(args.rng_seed)
    # Resume from an explicit --init-archive path, or from --out when --resume
    # is set; either way continue an existing front instead of re-seeding.
    init_path = args.init_archive or (
        args.out if (args.resume and os.path.exists(args.out)) else None)
    resuming = init_path is not None
    if resuming:
        archive = MapElitesArchive.load(init_path)
        dims = tuple(int(x) for x in archive.dims)  # int (not np.int64) for json log
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
                measures=aggregate_measures(ok, args.measures),
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

    # ---- seed the archive (skipped when resuming) ----
    if resuming:
        # Continue numbering past the last ATTEMPT (gen log records every
        # iteration, not just improving ones), so iters never duplicate.
        last_it = max([e.iteration for e in archive.cells.values()] + [0])
        if os.path.exists(gen_log_path):
            for line in open(gen_log_path):
                try:
                    it_n = json.loads(line).get("iteration")
                    if isinstance(it_n, int):
                        last_it = max(last_it, it_n)
                except Exception:
                    pass
        start_iter = last_it + 1
        print(f"  RESUMED from {init_path}: {len(archive.cells)} elites, "
              f"best_pop={archive.best.city_pop}, continuing at iter {start_iter}")
        log_gen({"event": "resume", "from": init_path,
                 "filled": len(archive.cells), "start_iter": start_iter})
    else:
        start_iter = 1
        for name in args.seeds.split(","):
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
                     "cell": list(cell), "fitness": r.fitness,
                     "measures": list(r.measures), "city_pop": r.city_pop,
                     "res_pop": r.res_pop, "com_pop": r.com_pop,
                     "ind_pop": r.ind_pop, "n_errors": r.n_errors,
                     "error": r.error, "eval_fits": spread, "source": SEEDS[name]})
            print(f"  seed {name:7s} fitness={r.fitness:8.1f} cityPop={r.city_pop:5d} "
                  f"cell={cell} rolls={spread} {'+' if imp else 'x'}")
        if not archive.cells:
            print("no seeds inserted — aborting."); return
        archive.record_history(0)

    # ---- evolve (optionally concurrent: --workers threads) ----
    # All shared state (archive, rng, gen_log, counters, stdout) is touched only
    # under `lock`; the slow parts (LLM call, env eval) run outside it, so N
    # workers overlap their LLM waits + eval bursts. MAP-Elites tolerates the
    # slightly-stale parent snapshots this implies (standard parallel QD).
    t0 = time.time()
    lock = threading.Lock()
    C = {"imp": 0, "inv": 0, "op": 0, "done": 0, "killed": 0}
    # Mod: discard fully open-loop genomes (needs the cl_ind reactivity descriptor).
    kill_open_loop = (args.measures == "cl_ind" and args.min_reactivity >= 0.0)
    if kill_open_loop:
        print(f"  killing fully open-loop genomes: "
              f"max(cf, divergence) <= {args.min_reactivity}")

    def run_one(it: int):
        try:
            with lock:
                do_cross = (len(archive.cells) >= 2
                            and rng.random() < args.crossover_rate)
                cl_demand = rng.random() < CL_DEMAND_PROB  # 50%: demand closed-loop
                origin = ("crossover" if do_cross else "mutate") + ("+cl" if cl_demand else "")
                if do_cross:
                    p1, p2 = archive.sample(rng, k=2, weighted=args.weighted_parents)
                    parents, renders, ps = (p1.eid, p2.eid), (p1.render, p2.render), (p1, p2)
                else:
                    (p,) = archive.sample(rng, k=1, weighted=args.weighted_parents)
                    parents, renders, ps = (p.eid,), p.render, (p,)
            # --- LLM operator (no lock) ---
            try:
                if do_cross:
                    directive = CLOSED_LOOP_DIRECTIVE if cl_demand else CROSSOVER_DIRECTIVE
                    child = operator.crossover(ps[0], ps[1], directive, renders=renders)
                else:
                    directive = CLOSED_LOOP_DIRECTIVE if cl_demand else MUTATE_DIRECTIVE
                    child = operator.mutate(ps[0], directive, render=renders)
            except OperatorError as e:
                with lock:
                    C["op"] += 1
                    log_gen({"event": "op_error", "iteration": it, "origin": origin,
                             "parents": list(parents), "error": str(e),
                             "source": getattr(e, "raw", None)})
                    print(f"it {it:4d} [{origin[:5]}] op-error: {e}")
                return
            # --- evaluation (no lock; spawns its own subprocesses) ---
            r, spread = evaluate(child)
            with lock:
                if not r.ok:
                    C["inv"] += 1
                    log_gen({"event": "invalid", "iteration": it, "origin": origin,
                             "parents": list(parents), "error": r.error, "source": child})
                    print(f"it {it:4d} [{origin[:5]}] invalid: {r.error}")
                    return
                # Mod: kill fully OPEN-LOOP genomes. With cl_ind, measures =
                # (cf_sensitivity, trajectory_divergence, ind_share); a policy
                # whose actions never change with the observation has cf≈div≈0.
                if kill_open_loop and max(r.measures[0], r.measures[1]) <= args.min_reactivity:
                    C["killed"] += 1
                    log_gen({"event": "open_loop_killed", "iteration": it,
                             "origin": origin, "parents": list(parents),
                             "cf_sensitivity": r.measures[0],
                             "trajectory_divergence": r.measures[1],
                             "city_pop": r.city_pop, "source": child})
                    print(f"it {it:4d} [{origin[:5]}] KILLED open-loop: "
                          f"cf={r.measures[0]:.2f} div={r.measures[1]:.2f} "
                          f"pop={r.city_pop}")
                    return
                imp, cell = archive.add(child, r.fitness, r.measures,
                                        city_pop=r.city_pop, parents=parents,
                                        origin=origin, iteration=it, render=r.render)
                log_gen({"event": "eval", "iteration": it, "origin": origin,
                         "parents": list(parents), "improved": bool(imp),
                         "cell": list(cell), "fitness": r.fitness,
                         "measures": list(r.measures), "city_pop": r.city_pop,
                         "res_pop": r.res_pop, "com_pop": r.com_pop,
                         "ind_pop": r.ind_pop, "n_errors": r.n_errors,
                         "error": r.error, "eval_fits": spread, "source": child})
                C["imp"] += int(imp)
                C["done"] += 1
                b = archive.best
                mstr = ",".join(f"{x:.2f}" for x in r.measures)
                print(f"it {it:4d} [{origin[:5]}] {'+' if imp else ' '} "
                      f"fit={r.fitness:8.1f} pop={r.city_pop:5d} m=({mstr}) "
                      f"cell={cell} | filled={len(archive.cells)} "
                      f"best_pop={b.city_pop} best_fit={b.fitness:.0f}")
                if C["done"] % args.save_every == 0:
                    archive.record_history(it)
                    archive.save(args.out)
        except Exception as e:  # noqa: BLE001 — never let one iteration kill the pool
            with lock:
                print(f"it {it:4d} UNEXPECTED: {type(e).__name__}: {e}")

    iters = list(range(start_iter, args.iters + 1))
    if args.workers <= 1:
        for it in iters:
            run_one(it)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, iters))

    archive.record_history(args.iters)
    archive.save(args.out)
    log_gen({"event": "run_end", "improved": C["imp"], "invalid": C["inv"],
             "op_errors": C["op"], "killed_open_loop": C["killed"],
             "filled": len(archive.cells)})
    gen_log.close()
    dt = time.time() - t0
    print("\n" + archive.summary())
    print(f"improved={C['imp']} invalid={C['inv']} op_errors={C['op']} "
          f"killed_open_loop={C['killed']} "
          f"in {dt:.1f}s ({dt/max(1,len(iters)):.2f}s/it, workers={args.workers})")
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
