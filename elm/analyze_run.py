"""Post-hoc analysis of an ELM (code-genome MAP-Elites) run.

Reads the saved `.npz` archive (and, if present, the `_generations.jsonl`
full trajectory) and reports:

  * coverage / QD-score / best, and the cityPop leaderboard;
  * a STATIC open-loop vs closed-loop classification of every elite's `act()`
    (does it react to obs.tile_map / live city metrics, or just replay a plan?);
  * feature flags per elite (plants, services, grids, masks, plan replay, ...);
  * a small set of strategy ARCHETYPES (feature-signature clusters) with a
    representative elite for each;
  * a behavior-space heatmap PNG (cityPop per res_ind cell).

Usage:
    python3 -m elm.analyze_run results/elm_diverse_overnight.npz \
        --png docs/elm_diverse_heatmap.png
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter, defaultdict

import numpy as np

# obs fields that mean the policy is REACTING to simulation state (closed loop),
# vs. fields that are just a clock (open loop can use these to index a plan).
FEEDBACK_FIELDS = {"tile_map", "city_pop", "res_pop", "com_pop", "ind_pop",
                   "funds", "powered_zones"}
CLOCK_FIELDS = {"step", "n_steps"}


def obs_fields_used(source: str) -> set:
    """All attribute names accessed on the `obs` parameter, via AST."""
    used = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return used
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "obs":
            used.add(node.attr)
    return used


def features(source: str) -> dict:
    """Static feature flags + the open/closed-loop verdict for one genome."""
    used = obs_fields_used(source)
    feedback = used & FEEDBACK_FIELDS
    s = source
    f = {
        "obs_fields": sorted(used),
        "reads_tilemap": "tile_map" in used,
        "reads_metrics": bool(feedback - {"tile_map"}),
        "closed_loop": bool(feedback),          # reacts to live sim state
        "uses_masks": any(m in s for m in ("wire_mask", "res_mask", "com_mask",
                          "ind_mask", "road_mask", "plant_mask", "empty_mask")),
        "uses_np_where": "np.where" in s or "np.argwhere" in s,
        "replays_plan": ('"plan"' in s or "'plan'" in s) and "state[" in s,
        "n_plants": s.count("COALPOWER") + s.count("NUCLEARPOWER"),
        "has_services": any(t in s for t in ("FIRESTATION", "POLICESTATION",
                            "STADIUM", "PARK", "SEAPORT", "AIRPORT")),
        "has_stadium": "STADIUM" in s,
        "has_rail": "RAILROAD" in s,
        "uses_com": "COMMERCIAL" in s,
        "uses_ind": "INDUSTRIAL" in s,
        "loc": source.count("\n") + 1,
    }
    return f


def archetype_signature(f: dict) -> str:
    """A coarse, human-readable strategy signature for clustering."""
    if f["closed_loop"]:
        kind = "CLOSED-LOOP/reactive"
    elif f["replays_plan"]:
        kind = "open-loop/plan-replay"
    else:
        kind = "open-loop/parametric"
    scale = ("multi-district" if f["n_plants"] >= 3 else
             "single-district" if f["n_plants"] == 1 else
             f"{f['n_plants']}-plant")
    svc = "+services" if f["has_services"] else "no-services"
    return f"{kind} | {scale} | {svc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--png", default=None, help="write a cityPop heatmap here")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    d = np.load(args.archive, allow_pickle=True)
    src = [str(x) for x in d["sources"]]
    fit = np.asarray(d["fitness"], float)
    cp = np.asarray(d["city_pop"], int)
    meas = np.asarray(d["measures"], float)
    org = [str(x) for x in d["origin"]]
    cells = np.asarray(d["cells"], int)
    dims = tuple(int(x) for x in d["dims"])
    n = len(src)

    feats = [features(s) for s in src]
    n_closed = sum(f["closed_loop"] for f in feats)

    print(f"# ELM run analysis — {args.archive}")
    print(f"elites={n}  coverage={n}/{int(np.prod(dims))}={n/np.prod(dims):.3f}"
          f"  qd_score={fit.clip(min=0).sum():.0f}")
    print(f"best cityPop={cp.max()}  best fitness={fit.max():.0f}")
    print(f"closed-loop elites={n_closed}/{n} ({100*n_closed/n:.0f}%)  "
          f"open-loop={n-n_closed}/{n}")
    print(f"origins: {dict(Counter(org))}")

    # leaderboard
    print(f"\n## cityPop leaderboard (top {args.top})")
    order = np.argsort(-cp)
    print(f"{'pop':>5} {'fit':>7} {'res':>4} {'ind':>4} {'orig':>9} "
          f"{'loop':>11} {'plants':>6} {'svc':>3}  signature")
    for i in order[:args.top]:
        f = feats[i]
        print(f"{cp[i]:5d} {fit[i]:7.0f} {meas[i,0]:4.2f} {meas[i,1]:4.2f} "
              f"{org[i]:>9} {'closed' if f['closed_loop'] else 'open':>11} "
              f"{f['n_plants']:6d} {'Y' if f['has_services'] else '-':>3}  "
              f"{archetype_signature(f)}")

    # archetype clustering by signature
    print("\n## strategy archetypes (by signature)")
    groups = defaultdict(list)
    for i, f in enumerate(feats):
        groups[archetype_signature(f)].append(i)
    for sig, idxs in sorted(groups.items(), key=lambda kv: -max(cp[i] for i in kv[1])):
        idxs = sorted(idxs, key=lambda i: -cp[i])
        rep = idxs[0]
        pops = cp[idxs]
        print(f"\n[{sig}]  n={len(idxs)}  best_pop={pops.max()} "
              f"median_pop={int(np.median(pops))}")
        print(f"    rep elite: cityPop={cp[rep]} fit={fit[rep]:.0f} "
              f"origin={org[rep]} m=({meas[rep,0]:.2f},{meas[rep,1]:.2f}) "
              f"obs_fields={feats[rep]['obs_fields']}")

    # closed-loop spotlight
    print("\n## closed-loop elites (react to obs)")
    cl = [i for i, f in enumerate(feats) if f["closed_loop"]]
    cl.sort(key=lambda i: -cp[i])
    for i in cl[:15]:
        print(f"  cityPop={cp[i]:5d} fit={fit[i]:7.0f} origin={org[i]:>9} "
              f"obs_fields={feats[i]['obs_fields']} masks={feats[i]['uses_masks']}")
    if not cl:
        print("  (none — every elite is open-loop)")

    # zone-mix diversity
    print("\n## behavior diversity (res_ind cells occupied)")
    rs, is_ = meas[:, 0], meas[:, 1]
    print(f"  res_share: min={rs.min():.2f} max={rs.max():.2f} "
          f"spread quartiles={np.percentile(rs,[25,50,75]).round(2).tolist()}")
    print(f"  ind_share: min={is_.min():.2f} max={is_.max():.2f} "
          f"spread quartiles={np.percentile(is_,[25,50,75]).round(2).tolist()}")

    # heatmap
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = np.full(dims, np.nan)
        for (c0, c1), pop in zip(cells, cp):
            grid[c0, c1] = max(grid[c0, c1], pop) if not np.isnan(grid[c0, c1]) else pop
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(grid.T, origin="lower", cmap="viridis",
                       extent=[0, 1, 0, 1], aspect="auto")
        ax.set_xlabel("res_share  (R / zoned tiles)")
        ax.set_ylabel("ind_share  (I / zoned tiles)")
        ax.set_title(f"ELM archive — cityPop per res_ind cell\n"
                     f"{n} elites, best={cp.max()}, {n_closed} closed-loop")
        fig.colorbar(im, ax=ax, label="cityPop")
        # mark the champion
        bi = int(np.argmax(cp))
        ax.scatter([meas[bi, 0]], [meas[bi, 1]], marker="*", s=260,
                   edgecolor="white", facecolor="red", zorder=5,
                   label=f"champion {cp[bi]}")
        ax.legend(loc="upper right", fontsize=8)
        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        fig.tight_layout(); fig.savefig(args.png, dpi=130)
        print(f"\nwrote heatmap -> {args.png}")


if __name__ == "__main__":
    main()
