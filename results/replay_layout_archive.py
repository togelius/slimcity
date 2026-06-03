#!/usr/bin/env python3
"""Replay-verify top-k elites from a layout (or any) .npz archive."""
from __future__ import annotations

import argparse
import numpy as np
from evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    for path in args.archive:
        d = np.load(path, allow_pickle=True)
        objs = d["objectives"]
        order = np.argsort(-objs)[: args.k]
        print(f"\n{path}  elites={len(objs)}  stored_max={objs.max():.1f}")
        best_pop = 0
        for rank, idx in enumerate(order, 1):
            theta = d["solutions"][idx].astype(np.float32)
            r = evaluate(
                theta,
                seed=42,
                n_actions=int(d["n_actions"]),
                ticks_per_action=int(d["ticks_per_action"]),
                warmup_ticks=int(d["warmup"]),
                policy_name=str(d["policy"]),
                policy_kwargs=eval(str(d["policy_kwargs"])),
                fitness_mode=str(d["fitness"]),
            )
            s = r.stats_final
            best_pop = max(best_pop, s.city_pop)
            print(
                f"  #{rank} stored={objs[idx]:.1f} replay={r.fitness:.1f} "
                f"cityPop={s.city_pop} R={s.res_pop} C={s.com_pop} I={s.ind_pop}"
            )
        print(f"  best replay cityPop in top-{args.k}: {best_pop}")


if __name__ == "__main__":
    main()
