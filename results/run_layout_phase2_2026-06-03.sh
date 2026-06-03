#!/bin/bash
# Phase 2: 200k-tick layout run + 500g warm-start from 50k archive.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/usr/bin/python3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMMON=(--policy layout --emitters 5 --batch 20 --workers 8
        --fitness dense --n-actions 1 --measures res_ind)

echo "=== $(date) layout 200k ticks, 200 gens ==="
$PY qd_train.py "${COMMON[@]}" --gens 200 --ticks-per-action 200000 \
  --archive-dims 20,20 \
  --save results/exp_2026-06-03_layout_ticks200k.npz

echo "=== $(date) layout continue 500g from 50k archive ==="
$PY qd_train.py "${COMMON[@]}" --gens 500 --ticks-per-action 100000 \
  --archive-dims 40,40 \
  --init-archive results/exp_2026-06-03_layout_resind_50k.npz \
  --save results/exp_2026-06-03_layout_resind_50k_cont500.npz

echo "=== $(date) replay top-5 ==="
$PY results/replay_layout_archive.py -k 5 \
  results/exp_2026-06-03_layout_ticks200k.npz \
  results/exp_2026-06-03_layout_resind_50k_cont500.npz \
  | tee results/exp_2026-06-03_layout_phase2_replay.log

echo "=== done $(date) ==="
