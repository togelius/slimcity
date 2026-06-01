"""Post-hoc analysis of LayoutGenome QD archives.

For each elite, decode the genome back into (zone grid, tax rate) and
correlate with fitness / measures. Answers questions like:
    - what zone mix do the best cities have?
    - does Micropolis reward low tax?
    - dense or sparse?

Usage:
    python3 analyze_layout.py archive_layout_resind.npz
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy import LayoutGenome, LAYOUT_CATEGORIES


CATEGORY_NAMES = ["empty", "residential", "commercial", "industrial", "park"]


def summarize(path: str) -> None:
    d = np.load(path, allow_pickle=True)
    sols = d["solutions"]
    objs = d["objectives"]
    meas = d["measures"]
    if len(objs) == 0:
        print("(empty archive)")
        return

    print(f"archive: {path}")
    print(f"  elites: {len(objs)}")
    print(f"  fitness range: [{objs.min():.1f}, {objs.max():.1f}]  mean: {objs.mean():.1f}")
    if "policy" in d.files:
        print(f"  policy: {d['policy']}")
    if "fitness" in d.files:
        print(f"  fitness mode: {d['fitness']}")
    if "measures_mode" in d.files:
        print(f"  measures: {d['measures_mode']}")

    # Decode each elite and pull category counts + tax
    proto = LayoutGenome()
    gh, gw = proto.GRID_H, proto.GRID_W
    n_cells = gh * gw

    cat_counts = np.zeros((len(objs), proto.N_CAT), dtype=np.int32)
    taxes = np.zeros(len(objs), dtype=np.int32)
    for i, theta in enumerate(sols):
        proto.set_params(theta.astype(np.float32))
        grid, tax = proto.decode()
        for c in range(proto.N_CAT):
            cat_counts[i, c] = int((grid == c).sum())
        taxes[i] = tax

    cat_shares = cat_counts / n_cells

    # Sort elites by fitness, descending
    order = np.argsort(objs)[::-1]
    print("\n  top 10 elites by fitness:")
    print(f"  {'rank':>4}  {'fit':>7}  {'tax':>3}  {'empty':>5}  {'R':>4}  {'C':>4}  {'I':>4}  {'park':>4}  measures")
    for rank, idx in enumerate(order[:10], 1):
        print(f"  {rank:>4}  {objs[idx]:>7.1f}  {taxes[idx]:>3}  "
              f"{cat_shares[idx, 0]:>5.2f}  "
              f"{cat_shares[idx, 1]:>4.2f}  {cat_shares[idx, 2]:>4.2f}  "
              f"{cat_shares[idx, 3]:>4.2f}  {cat_shares[idx, 4]:>4.2f}  "
              f"({meas[idx, 0]:.2f}, {meas[idx, 1]:.2f})")

    # Tax distribution
    print(f"\n  tax distribution across archive:")
    print(f"    min={taxes.min()}  max={taxes.max()}  mean={taxes.mean():.1f}  median={int(np.median(taxes))}")
    # Bin and show
    for lo, hi in [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
        mask = (taxes >= lo) & (taxes <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_fit = float(objs[mask].mean())
        print(f"    tax {lo:2d}..{hi:2d}: {n:3d} elites, mean fitness {avg_fit:.1f}")

    # Mean category shares of top-10 elites vs all
    top10_share = cat_shares[order[:10]].mean(axis=0)
    all_share   = cat_shares.mean(axis=0)
    print(f"\n  mean category share — top 10 vs whole archive:")
    print(f"  {'category':>12}  {'top10':>6}  {'all':>6}")
    for c, name in enumerate(CATEGORY_NAMES):
        print(f"  {name:>12}  {top10_share[c]:>6.2f}  {all_share[c]:>6.2f}")

    # Tax of best
    best_idx = int(np.argmax(objs))
    print(f"\n  BEST ELITE: fitness={objs[best_idx]:.1f}  tax={taxes[best_idx]}  measures=({meas[best_idx,0]:.2f}, {meas[best_idx,1]:.2f})")
    # Print the grid layout of the best (compact ASCII)
    proto.set_params(sols[best_idx].astype(np.float32))
    grid, _ = proto.decode()
    chars = ['.', 'R', 'C', 'I', 'p']
    print("\n  best city layout (rows are top->bottom, cols left->right):")
    for j in range(gh):
        print("   ", "".join(chars[int(grid[j, i])] for i in range(gw)))


if __name__ == "__main__":
    paths = sys.argv[1:] or ["archive_layout_resind.npz"]
    for p in paths:
        summarize(p)
        print()
