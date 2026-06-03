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

## 2026-05-31 (later) — Growth fitness + RandomPrefix

Hypothesis: the variety bonus saturates too quickly to give the net half
useful gradient. A denser `growth` fitness rewards "your zone just
transitioned past ungrown" and "your zone is adjacent to a road".
RandomPrefix replaces the evolved tape with a *random* one per eval,
combined with multi-eval averaging for noise robustness.

| config                          | evals | wall  | best obj | cityPop | insight |
| ------------------------------- | ----: | ---:  | -------: | ------: | ------- |
| `deepconv` (16,32) `growth`     | 3,000 | 1253s |   112.0  |       0 | Growth fitness alone didn't unblock deepconv. Plateau at obj~50 after gen 3. |
| `randprefix` 50+(16,32) ×5      | 2,500 | 8396s |   146.5  |     160 | **First net-based growth: 1 industrial zone.** ~140 min wall time. |
| `randprefix` 50+(16,32) ×10     | 1,200 | 7977s |    82.3  |       0 | Heavier smoothing + fewer gens didn't reach growth. |
| `tape` @200 `growth`            | 3,000 |  525s |   686.0  |     640 | Growth fitness preserved tape's performance. |

## 2026-06-01 — Ambitious overnight batch (40×40 archive)

Run with persistent .npz metadata and a 1600-cell archive for richer
diversity. Total wall time: 7h 39min.

| config                          | evals  | wall   | best obj | cityPop | R/C/I | insight |
| ------------------------------- | -----: | -----: | -------: | ------: | ----- | ------- |
| **`tape` @600 varied**          |  5,000 |  2603s | **1,202.2** | **1,120** | **8/0/5** | **Breakthrough: first mixed R+I city.** 8 residential + 5 industrial. The longer rollout (60k ticks ≈ 30 in-game years) gives residential demand time to develop after industry. |
| `tape` @1000 varied             |  3,000 |  2694s |  1,248.6 |     640 | 0/0/5 | Regression: bigger isn't always better. Replay (768) doesn't match stored (1248) — small residual non-determinism. |
| `deepconv` (16,32) growth 150g  | 15,000 |  6138s |    119.8 |       0 | 0/0/0 | 347/1600 archive cells filled — wide diversity of *failure modes*. Long budget didn't help. |
| `randprefix` ×5 growth 25g      | 12,500 |  8146s |     85.8 |     160 | 0/0/1 | Reproduces 1 ind zone but lower obj_max than May 31 5x run (different gen budget). |
| `randprefix` ×10 growth 12g     |  6,000 |  7922s |     99.2 |       0 | 0/0/0 | Only 9 archive cells. Too few gens after 10× cost increase. |

Saved archives + heatmaps in `results/overnight_*.{npz,png}`. Best elite
GIF at `results/tape_600_best.gif` — open in macOS Preview.

## 2026-06-01 — Richer QD measures + longer runs (40×40 archive)

Added `--measures {road_ind, res_ind, density}` so QD descriptors aren't
locked to the original (road_frac, ind_share) pair. Also fixed two engine
zero-init bugs (calloc in newPtr; Map._mapData fill in constructor) that
partially address the residual non-determinism — same θ now gives a
tighter spread of fitness values across processes, but still not bit-
identical. Pragmatic workaround: `--n-evals N` to average fitness across
N rollouts.

| config                                | evals  | wall   | best obj | cityPop | R/C/I | insight |
| ------------------------------------- | -----: | -----: | -------: | ------: | ----- | ------- |
| `tape@800` res_ind n_evals=3          | 15,000 | 10780s |  1,096.3 |    780 | 11/0/4 | Most residential ever (11 R), but lower total than tape@600 — different city shape. |
| `tape@600` res_ind n_evals=1          |  5,000 |  2590s |  1,202.2 |  1,120 |  8/0/5 | Same peak as overnight road_ind; res_ind broadens elites 6× (33 → 196). |
| `hybrid` 80+(16,32)@250 growth n=3    | 12,000 |  9221s |    733.2 |    640 |  0/0/5 | Highest archive coverage (483 elites), peak unchanged. |
| `randprefix` 5× 40g density growth    | 40,000 | 13169s |    152.8 |      0 |  -    | Density measures clustered all elites into 1 cell — bad measure choice. |
| `deepconv` 200g density growth n=3    | 60,000 | 24835s |    144.7 |      0 |  -    | Same — density doesn't discriminate net policies. |

## 2026-06-02 — Perturbation study

Investigated why `hybrid` has 3× more archive elites than `tape@600`
despite tape having 1.8× higher peak. Took the best elite of each, then
generated 60 random θ perturbations at six σ levels and re-evaluated.

| policy        | peak | σ=0.01 median | σ=0.01 max | distinct cells σ=1.0 |
| ------------- | ---: | ------------: | ---------: | -------------------: |
| `tape@600`    | 1202 |          74.3 |      871.2 |                   33 |
| `hybrid_long` |  684 |          33.2 |      681.0 |                   52 |

Findings:
- **Tape's peak is a knife-edge.** σ=0.010 perturbation drops median fitness 94%. Almost no "good neighbors" exist in θ-space.
- **Hybrid is equally sharp** — its peak is also surrounded by low-fitness perturbations.
- **Hybrid spreads more measures-cells per perturbation** because it has 6× more parameters, each contributing an independent measure-space direction.
- **The hybrid archive isn't full of better cells than tape's** — it's full of *more* cells, mostly low fitness.
- Even σ=1.0 perturbations of tape occasionally hit fitness 870+ (cityPop ~830) — good policies exist scattered across θ-space, but the QD measures don't put them adjacent to the peak.

Takeaway: the QD measure choice strongly affects what "neighbor" means.
With richer measures (e.g. discretized R/C/I shares), tape might fill
more cells without changing the peak. The non-trivial insight is that
adjacency in θ-space does NOT predict adjacency in measure-space.

## 2026-06-02 — Long randprefix runs with n_evals=10

Hypothesis: averaging fitness over 10 random scaffold seeds gives CMA-ES
a smoother gradient, leading to more robust policies. Three configs.

| config                                | evals  | wall   | best obj | cityPop | insight |
| ------------------------------------- | -----: | -----: | -------: | ------: | ------- |
| `randprefix` 50+(16,32)@200 n=10 25g  |  5,000 | 16916s |     82.9 |       0 | Worse than n=5 — 10x averaging kills the lucky-seed strategy that previously hit cityPop=160. |
| `randprefix` 30+(8,16)@300 n=10 25g   |  5,000 | 15784s |     80.5 |       0 | Same — averaging selects for robustness over peak performance. |
| `randprefix` 100+(16,32)@400 n=10 15g |  3,000 | (~6h)  |   pending| pending | Bigger prefix, longer rollout. |

Surprising finding: **heavier averaging hurts on a rare-success
landscape.** When the underlying policy works only 1 in N attempts,
averaging over more attempts dilutes the signal. n=5 was probably the
sweet spot. The next-up runs should drop back to n=3 or n=5.

## 2026-06-02 — Rich observations for closed-loop nets

Hypothesis: the 4-channel binary obs is too impoverished. Built
`RichDeepConvPolicy` with 14-channel obs:
  - 6 spatial channels: R, C, I, infra, **power**, **growth**
  - 8 broadcast scalars: cityPop, funds, step_frac, pollution, crime,
    R/C/I demands (from `engine.getDemands()`)

12,452 params at `channels=(16,32)` — only 14% more than basic DeepConv.

| config                                    | evals | wall   | best obj | cityPop | elites | insight |
| ----------------------------------------- | ----: | -----: | -------: | ------: | -----: | ------- |
| `rich_deepconv` (16,32) growth 60g res_ind|  6,000 |  7898s |    118.9 |       0 |    353 | **Rich obs alone doesn't unlock growth.** But archive coverage 6× larger than basic deepconv. |
| `rich_deepconv` (16,32) varied 40g res_ind|  4,000 |  4968s |    104.2 |       0 |    181 | Simpler fitness, same story. |
| `rich_deepconv` (8,16) growth 80g res_ind | 8,000 | pending| pending  | pending | pending| Smaller net + more gens. |

Verdict: richer obs helps the policy be **more diverse** (4–6× more
archive cells filled than basic deepconv) but still can't grow a single
zone. So obs is **not** the only bottleneck — the closed-loop nets still
fail at the joint coordination problem, even with full visibility.

Remaining suspects:
- Zero-init argmax degeneracy is still there at θ=0
- Single-tile placement may be too granular
- The fitness landscape's "find 4 adjacent things at once" structure
  defeats gradient-based search regardless of obs richness

Next: try **entropy_count** as a *curriculum* descriptor that nudges
search toward "diverse, dense" — the regime where viable cities live.
Queued via LaunchAgent for after current batches finish.

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
