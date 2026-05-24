"""Print a text-art visualization of the QD archive.

Usage:
    python3 plot_archive.py [archive.npz]
"""

from __future__ import annotations

import sys

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "archive.npz"
    d = np.load(path)
    objectives = d["objectives"]
    measures = d["measures"]
    if len(objectives) == 0:
        print("(empty archive)")
        return

    print(f"archive: {len(objectives)} elites")
    print(f"  obj_max={objectives.max():.1f}  obj_min={objectives.min():.1f}  "
          f"obj_mean={objectives.mean():.1f}")
    print(f"  measure 0 (road_frac): [{measures[:,0].min():.2f} .. {measures[:,0].max():.2f}]")
    print(f"  measure 1 (ind_share): [{measures[:,1].min():.2f} .. {measures[:,1].max():.2f}]")

    # 20x20 text-art grid
    H, W = 20, 20
    grid = np.full((H, W), -np.inf, dtype=np.float32)
    for o, m in zip(objectives, measures):
        i = min(int(m[0] * W), W - 1)   # road_frac column
        j = min(int(m[1] * H), H - 1)   # ind_share row
        if o > grid[j, i]:
            grid[j, i] = o

    print("\n  rows=ind_share (top=high), cols=road_frac (left=0, right=1)")
    print("  '.' empty, '#' filled (label = best fitness)")
    for j in range(H - 1, -1, -1):
        line = ["    "]
        for i in range(W):
            if grid[j, i] == -np.inf:
                line.append(" .  ")
            else:
                line.append(f"{int(grid[j,i]):4d}")
        print("".join(line))


if __name__ == "__main__":
    main()
