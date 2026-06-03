#!/bin/bash
# Long layout evolution batch (2026-06-03, M4). Logs to results/*.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/usr/bin/python3
export PATH="/opt/homebrew/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== layout 50k evals (100k ticks, 40x40 archive) ===" | tee results/exp_2026-06-03_layout_resind_50k.log
$PY qd_train.py --policy layout --gens 500 --emitters 5 --batch 20 \
  --workers 8 --fitness dense --n-actions 1 --ticks-per-action 100000 \
  --measures res_ind --archive-dims 40,40 \
  --save results/exp_2026-06-03_layout_resind_50k.npz \
  2>&1 | tee -a results/exp_2026-06-03_layout_resind_50k.log

echo "=== layout 20k evals (200k ticks) ===" | tee results/exp_2026-06-03_layout_ticks200k.log
$PY qd_train.py --policy layout --gens 200 --emitters 5 --batch 20 \
  --workers 8 --fitness dense --n-actions 1 --ticks-per-action 200000 \
  --measures res_ind --archive-dims 20,20 \
  --save results/exp_2026-06-03_layout_ticks200k.npz \
  2>&1 | tee -a results/exp_2026-06-03_layout_ticks200k.log

echo "=== replay top-5 from each archive ===" | tee results/exp_2026-06-03_layout_replay.log
$PY - <<'PY' 2>&1 | tee -a results/exp_2026-06-03_layout_replay.log
import numpy as np
from evaluate import evaluate

def replay(path, k=5):
    d = np.load(path, allow_pickle=True)
    objs = d["objectives"]
    order = np.argsort(-objs)[:k]
    print(f"\n{path}  elites={len(objs)}  stored_max={objs.max():.1f}")
    for rank, idx in enumerate(order, 1):
        theta = d["solutions"][idx].astype(np.float32)
        r = evaluate(
            theta, seed=42,
            n_actions=int(d["n_actions"]),
            ticks_per_action=int(d["ticks_per_action"]),
            warmup_ticks=int(d["warmup"]),
            policy_name=str(d["policy"]),
            policy_kwargs=eval(str(d["policy_kwargs"])),
            fitness_mode=str(d["fitness"]),
        )
        s = r.stats_final
        print(f"  #{rank} stored={objs[idx]:.1f} replay={r.fitness:.1f} "
              f"cityPop={s.city_pop} R={s.res_pop} C={s.com_pop} I={s.ind_pop}")

for p in (
    "results/exp_2026-06-03_layout_resind_50k.npz",
    "results/exp_2026-06-03_layout_ticks200k.npz",
):
    replay(p)
PY
echo "=== done ==="
