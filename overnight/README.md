# Overnight ELM run (1000 actions × 10 ticks)

Evolves closed-loop tile-placement policies with Claude as the mutation/crossover
operator, seeded with the best closed-loop strategies found so far (the scalable
serviced comb `comb1000` ~83k, plus `serviced`, `clind_react`, `clind_long`,
`clind_fine`) and the diverse hand-written founders.

## Launch (needs an Anthropic key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # the Claude operator reads this
zsh overnight/launch_overnight.sh             # starts the run, writes overnight/elm.pid

# watchdog: emails julian@togelius.com + commits/pushes results every 45 min
PYTHONPATH=. nohup /usr/bin/python3 overnight/monitor_overnight.py \
    --log results/elm_comb1000_overnight.log \
    --npz results/elm_comb1000_overnight.npz \
    --pid-file overnight/elm.pid --interval 45 \
    > overnight/monitor.log 2>&1 &
```

## Inspect / stop

```bash
tail -f results/elm_comb1000_overnight.log            # live progress
kill $(cat overnight/elm.pid)                         # stop the run (monitor sends a final email)
```

Config: `cl_ind` measures (closed-loopness), 8×8×10 archive, `dense` fitness,
n_evals=3, workers=6, per-eval timeout 60 s. Results land in
`results/elm_comb1000_overnight.npz` (+ `_best.py`), committed to `main`.
