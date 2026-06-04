#!/bin/bash
# Babysits the weekend layout run: every hour, regenerate a progress heatmap +
# stats snapshot and commit them (Dropbox syncs the repo across machines; git
# push is attempted best-effort but may fail if no non-interactive credential).
# Every 6h also commits the full .npz archive. Stops the run Monday >= 06:00.
cd "$(dirname "$0")"
ARCH=results/weekend_layout_resind.npz
LOG=results/weekend_layout_resind.log
PNG=docs/weekend_progress.png
TXT=results/weekend_progress.txt
iter=0
while true; do
  sleep 3600
  iter=$((iter+1))

  # progress stats snapshot
  /usr/bin/python3 - "$ARCH" > "$TXT" 2>&1 <<'PY'
import sys, numpy as np, time
try:
    d = np.load(sys.argv[1], allow_pickle=True)
    o = d["objectives"]
    print(f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"elites: {len(o)} / {int(np.prod(d['grid_dims']))} cells")
    print(f"obj_max={o.max():.1f}  obj_mean={o.mean():.1f}  qd_score={o[o>0].sum():.0f}")
    print(f"gens configured: {int(d['gens'])}  wall_seconds={float(d['wall_seconds']):.0f}")
except Exception as e:
    print("no archive yet:", e)
PY

  # heatmap (best-effort)
  /usr/bin/python3 plot_archive_png.py "$ARCH" "$PNG" --title "weekend layout (live)" --auto-zoom >/dev/null 2>&1 || true

  # commit small artifacts hourly; full archive every 6h
  git add "$LOG" "$TXT" 2>/dev/null
  git add -f "$PNG" 2>/dev/null
  if [ $((iter % 6)) -eq 0 ]; then git add -f "$ARCH" 2>/dev/null; fi
  git commit -m "weekend run: progress checkpoint $(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1 || true
  GIT_TERMINAL_PROMPT=0 git push origin main >/dev/null 2>&1 || true

  # Monday >= 06:00 -> final checkpoint, stop the run, exit
  if [ "$(date +%u)" -eq 1 ] && [ "$(date +%H)" -ge 6 ]; then
    pkill -f "qd_train.py --policy layout" 2>/dev/null
    sleep 10
    git add -A 2>/dev/null
    git add -f "$ARCH" "$PNG" 2>/dev/null
    git commit -m "weekend run: FINAL checkpoint $(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1 || true
    GIT_TERMINAL_PROMPT=0 git push origin main >/dev/null 2>&1 || true
    break
  fi
done
