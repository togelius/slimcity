"""Perturbation study: how isolated is the peak elite?

For each policy under study, take the best elite, perturb its θ at several
σ levels, evaluate each perturbation, and report:
  - how fitness degrades vs σ
  - how many distinct measure-space cells the perturbations hit

This tells us whether the peak is a sharp spike (few neighbors, hard for
CMA-ES to fill nearby cells) or a broad plateau.

Usage:
    python3 perturb_study.py <archive.npz> [--n 100] [--sigmas 0.01,0.05,0.1,0.3]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np

from evaluate import evaluate


_W = None


def _init(kwargs: dict) -> None:
    global _W
    _W = kwargs


def _eval(theta: np.ndarray) -> tuple[float, float, float]:
    r = evaluate(theta, **_W)
    return float(r.fitness), float(r.measures[0]), float(r.measures[1])


def _kwargs_from_archive(d: dict) -> dict:
    pol = str(d["policy"])
    kw = eval(str(d["policy_kwargs"]))
    measures_mode = str(d["measures_mode"]) if "measures_mode" in d.files else "road_ind"
    return dict(
        seed=42,
        n_actions=int(d["n_actions"]),
        ticks_per_action=int(d["ticks_per_action"]),
        warmup_ticks=int(d["warmup"]),
        policy_name=pol,
        policy_kwargs=kw,
        fitness_mode=str(d["fitness"]),
        measures_mode=measures_mode,
        n_evals=1,  # single eval per perturbation for speed
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", help=".npz archive to study")
    ap.add_argument("--n", type=int, default=100,
                    help="perturbations per sigma")
    ap.add_argument("--sigmas", type=str, default="0.0,0.01,0.03,0.1,0.3,1.0",
                    help="comma-separated sigma values to test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="rng seed for perturbations")
    ap.add_argument("--out", type=str, default=None,
                    help="optional .json to dump full results")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]

    d = np.load(args.archive, allow_pickle=True)
    objs = d["objectives"]
    idx = int(np.argmax(objs))
    theta0 = d["solutions"][idx].astype(np.float32)
    n_params = theta0.size
    print(f"archive: {args.archive}")
    print(f"  policy = {str(d['policy'])}  measures = {str(d['measures_mode']) if 'measures_mode' in d.files else 'road_ind'}")
    print(f"  best elite: idx={idx}, stored obj={objs[idx]:.1f}, "
          f"measures=({d['measures'][idx][0]:.3f}, {d['measures'][idx][1]:.3f}), "
          f"params={n_params}")

    eval_kwargs = _kwargs_from_archive(d)

    # Bake the perturbation vectors for reproducibility
    rng = np.random.default_rng(args.seed)
    thetas_by_sigma: list[tuple[float, np.ndarray]] = []
    for sigma in sigmas:
        # 1 baseline (sigma=0) + n-1 perturbations
        n_real = 1 if sigma == 0 else args.n
        for k in range(n_real):
            if sigma == 0 and k == 0:
                thetas_by_sigma.append((sigma, theta0.copy()))
            else:
                noise = rng.standard_normal(n_params).astype(np.float32) * sigma
                thetas_by_sigma.append((sigma, theta0 + noise))

    flat_thetas = [t for _, t in thetas_by_sigma]
    sigmas_list = [s for s, _ in thetas_by_sigma]

    print(f"\n{len(flat_thetas)} evaluations across {len(sigmas)} sigma levels "
          f"({args.workers} workers)...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(eval_kwargs,)) as pool:
        results = pool.map(_eval, flat_thetas)

    # Aggregate per sigma
    per_sigma: dict[float, dict[str, Any]] = {}
    for sigma in sigmas:
        per_sigma[sigma] = {"fitnesses": [], "measures": []}
    for (sigma, _), (f, m0, m1) in zip(thetas_by_sigma, results):
        per_sigma[sigma]["fitnesses"].append(f)
        per_sigma[sigma]["measures"].append((m0, m1))

    # Grid cells
    grid = (int(d["grid_dims"][0]), int(d["grid_dims"][1])) if "grid_dims" in d.files else (20, 20)
    gw, gh = grid

    def cell_of(m0: float, m1: float) -> tuple[int, int]:
        return (min(int(m0 * gw), gw - 1), min(int(m1 * gh), gh - 1))

    print(f"\n{'sigma':>8s}  {'n':>4s}  {'fit_mean':>9s}  {'fit_med':>9s}  "
          f"{'fit_min':>9s}  {'fit_max':>9s}  {'distinct_cells':>14s}  "
          f"{'frac_at_peak':>12s}")
    peak_obj = per_sigma[0.0]["fitnesses"][0] if 0.0 in per_sigma and per_sigma[0.0]["fitnesses"] else float(objs[idx])
    for sigma in sigmas:
        fits = per_sigma[sigma]["fitnesses"]
        meas = per_sigma[sigma]["measures"]
        if not fits:
            continue
        cells = {cell_of(m0, m1) for m0, m1 in meas}
        frac_at_peak = sum(1 for f in fits if abs(f - peak_obj) < 1.0) / len(fits)
        print(f"{sigma:8.3f}  {len(fits):4d}  {np.mean(fits):9.1f}  "
              f"{np.median(fits):9.1f}  {np.min(fits):9.1f}  {np.max(fits):9.1f}  "
              f"{len(cells):14d}  {frac_at_peak:12.2f}")

    if args.out:
        payload = {
            "archive": args.archive,
            "policy": str(d["policy"]),
            "best_idx": idx,
            "best_obj": float(objs[idx]),
            "best_measures": [float(d["measures"][idx][0]), float(d["measures"][idx][1])],
            "n_params": int(n_params),
            "per_sigma": {
                str(s): {
                    "fitnesses": v["fitnesses"],
                    "measures": v["measures"],
                } for s, v in per_sigma.items()
            },
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote full data to {args.out}")


if __name__ == "__main__":
    main()
