# Experiment log

Chronological log of training runs. For "what works / what doesn't" see
`RESULTS.md`. Archives live under `results/` (gitignored — regenerate
from the recipes below).

The columns:
- **config** — short policy + key knobs
- **evals** — total `gens × emitters × batch` (× `n_evals` if averaged)
- **wall time** — on the M4 with 8 workers
- **best** — `obj_max` and `cityPop` of best replay-verified elite
- **insight** — what this run taught us

---

## 2026-05-23 — 1000-eval smoke (broken-RNG era)

| config                    | evals  | wall | best obj | cityPop | insight |
| ------------------------- | -----: | ---: | -------: | ------: | ------- |
| `conv`, `pop` fitness     |  1,000 | 8 min |   0.0   |       0 | Hard task: cityPop alone is too sparse. 23 archive cells filled from tile-pattern variety, all at fitness 0. |

## 2026-05-24 — Policy zoo bench

| config                  | evals | wall  | best obj | cityPop | insight |
| ----------------------- | ----: | ---:  | -------: | ------: | ------- |
| `tape` @100, `dense`    | 1,000 | 165s  |  333.6\* |    640\*| Direct (x,y,tool) coords beat networks by 10×. (*Pre-determinism era — replays were noisy; the 665 was a lucky roll.) |
| `conv` @100, `dense`    | 1,000 | 157s  |   44.1   |       0 | Closed-loop net stuck in zero-init argmax degeneracy. |
| `deepconv` (16,32)      | 1,000 | 459s  |   24.4   |       0 | Adding depth slows training, doesn't help. |
| `mlp` (hidden=32)       | 1,000 |  99s  |    4.6   |       0 | Too many params (~78k) for the budget. |

## 2026-05-25 — Dinner experiments

| config                          | evals | best obj | cityPop | insight |
| ------------------------------- | ----: | -------: | ------: | ------- |
| `tape` @100, `varied`           | 5,000 |  665.1\* |  varied | Variety bonus boosted tape but replay was non-deterministic — exposed the engine RNG bug. |
| `ctxtape` @100, `varied`        | 5,000 |  181.9   |       0 | Closed-loop tape doesn't beat open-loop. |
| `conv` @100, `varied`           | 3,000 |  118.0   |       0 | Variety bonus broke degeneracy but no real growth. |

## 2026-05-26 — Engine determinism patch

**Fix:** removed `randomlySeedRandom()` call from `Micropolis::initWillStuff()`
([engine/src/initialize.cpp:78](engine/src/initialize.cpp)). It was reseeding
the LCG from `gettimeofday()` after our deterministic `generateMap(seed)`,
so same θ + same `env.reset(seed=42)` gave different fitnesses across Python
processes.

After rebuild, 5/5 separate invocations of the best tape elite return
identical fitness=185.1 / cityPop=160. The pre-fix 665 was a lucky non-
determinism roll.

## 2026-05-30 — Longer tape + deeper nets (deterministic engine)

| config                          | evals | wall  | best obj | cityPop | insight |
| ------------------------------- | ----: | ---:  | -------: | ------: | ------- |
| `tape` @100, `varied` (re-run)  | 5,000 |  438s |   508.9  |     480 | 3 industrial zones. Deterministic ground truth. |
| `tape` @200, `varied`           | 5,000 |  536s |   681.8  |     640 | 4 industrial zones. ~1 zone per +100 actions. |
| **`tape` @400, `varied`**       | 5,000 |  892s | **849.3**| **800** | 5 industrial zones. Best result so far. |
| `deepconv` (16,32,16) `varied`  | 3,000 | 1849s |   117.7  |       0 | 3rd conv layer doesn't help. |
| `deepconv` (32,64) `varied`     | 4,000 | 4017s |    63.3  |       0 | Wider net is WORSE; 31k params overwhelm CMA-ES. |
| `mlp` (hidden=64) `varied`      | 2,000 |  286s |    11.9  |       0 | 155k params hopeless in 2k evals. |

## 2026-05-31 — Hybrid (tape prefix + DeepConv tail)

Idea: tape solves bootstrap by carpet-bombing, then the closed-loop net
refines on the partially-built map.

| config                          | evals | best obj | cityPop | insight |
| ------------------------------- | ----: | -------: | ------: | ------- |
| `hybrid` tape50+(8,16) @200     | 5,000 |  671.2   |    640  | 4 zones — matches tape@200, doesn't exceed. |
| `hybrid` tape50+(16,32) @200    | 4,000 |  663.5   |    640  | Bigger net doesn't help. |
| `hybrid` tape100+(8,16) @300    | 4,000 |  679.7   |    640  | Longer tape doesn't help. |

The net half generates **diversity** (107–114 archive cells filled vs
13 for tape@200) but **no extra zones**. The tape half does all the
heavy lifting.

## 2026-05-31 (later) — Growth fitness + RandomPrefix [IN PROGRESS]

Hypothesis: the variety bonus saturates too quickly to give the net half
of hybrid useful gradient. A denser `growth` fitness rewards "your zone
just transitioned past ungrown" and "your zone is adjacent to a road".
RandomPrefix replaces the evolved tape with a *random* one per eval,
combined with multi-eval averaging for noise robustness.

| config | evals | status |
| ------ | ----: | ------ |
| `deepconv` (16,32) `growth`           | 3,000 | running |
| `randprefix` 50+(16,32) `growth` ×5   | 2,500 | queued |
| `randprefix` 50+(16,32) `growth` ×10  | 1,200 | queued |
| `tape` @200 `growth`                  | 3,000 | queued |

## How to add a row

When you launch a long run, append a row here with the same shape.
Replay-verified `cityPop` is the important number; `obj_max` alone can
look good but be all bonus.

```python
# Replay verification snippet
import numpy as np
from evaluate import evaluate
d = np.load('results/<name>.npz', allow_pickle=True)
idx = int(np.argmax(d['objectives']))
theta = d['solutions'][idx].astype(np.float32)
r = evaluate(theta, seed=42,
             n_actions=int(d['n_actions']),
             ticks_per_action=int(d['ticks_per_action']),
             warmup_ticks=int(d['warmup']),
             policy_name=str(d['policy']),
             policy_kwargs=eval(str(d['policy_kwargs'])),
             fitness_mode=str(d['fitness']))
print(r.fitness, r.stats_final.city_pop,
      r.stats_final.res_pop, r.stats_final.com_pop, r.stats_final.ind_pop)
```
