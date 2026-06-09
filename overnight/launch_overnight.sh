#!/bin/zsh
# Launch the overnight ELM run for the 1000x10 episode, seeded with the top
# closed-loop strategies (champions) + the diverse hand-written founders.
#
# Requires ANTHROPIC_API_KEY in the environment (the Claude operator reads it).
# Run from the repo root:  zsh overnight/launch_overnight.sh
set -e
cd "$(dirname "$0")/.."          # repo root
ROOT="$(pwd)"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set — the Claude operator needs it." >&2
    exit 1
fi

OUT="results/elm_comb1000_overnight.npz"
LOG="results/elm_comb1000_overnight.log"

# Champion seeds (top closed-loop strategies) + diverse founders for behavior spread.
SEEDS="comb1000,serviced,clind_react,clind_long,clind_fine,plan,industrial,commercial,mixed,res_ind,res_heavy,grid,twin,reactive,reactive_wire"

echo "Launching ELM @1000x10 -> $LOG"
PYTHONPATH="$ROOT" nohup /usr/bin/python3 -m elm.elm_train \
    --operator claude --model claude-sonnet-4-6 --temperature 1.0 \
    --seeds "$SEEDS" \
    --n-actions 1000 --ticks-per-action 10 \
    --measures cl_ind --archive-dims 8,8,10 --fitness dense \
    --n-evals 3 --workers 6 --timeout 60 \
    --iters 100000 --save-every 20 \
    --out "$OUT" \
    > "$LOG" 2>&1 &

echo $! > overnight/elm.pid
echo "ELM PID $(cat overnight/elm.pid) — tail -f $LOG"
