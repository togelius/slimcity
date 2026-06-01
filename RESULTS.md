# slimcity — what we know so far

A compact "what works / what doesn't" you can read in 2 minutes before
deciding what to try.

## Headline

Best deterministic city grown from a blank map (as of the latest batch):

| metric                             | best |
| ---------------------------------- | ---- |
| **cityPop**                        | **800** |
| number of industrial zones grown   | 5    |
| policy                             | `tape` with 400 actions × 100 ticks |
| total params                       | 1,200 (just (tool, x, y) triples) |
| fitness mode                       | `varied` |

Best `cityPop` per representation (with the deterministic engine):

| representation        | best cityPop | obj_max | params  | notes |
| --------------------- | -----------: | ------: | ------: | ----- |
| `tape` @100           |          480 |   508.9 |    300  | 3 ind. zones |
| `tape` @200           |          640 |   681.8 |    600  | 4 ind. zones |
| **`tape` @400**       |      **800** |   849.3 |  1,200  | 5 ind. zones |
| `hybrid` (tape+conv)  |          640 |   679.7 |  ~5–11k | matches tape, doesn't beat |
| `ctxtape`             |            0 |   181.9 |    330  | bonuses only |
| `conv`                |            0 |   118.0 |    740  | bonuses only |
| `deepconv` (16,32)    |            0 |   118.0 | 11–13k  | bonuses only |
| `deepconv` (32,64)    |            0 |    63.3 | 31k     | bonuses only — got worse with more params |
| `mlp` (h=32 / h=64)   |            0 |   10–12 | 78–155k | bonuses only — too many params for eval budget |

Only `tape` and `hybrid` have produced cityPop > 0. Everything else has
been gaming the variety/dense bonuses without growing zones.

## What works

1. **Open-loop `ActionTape`** is the only policy reliably growing cities.
   Each tape entry is `(tool, x, y)` tanh-mapped to a discrete tile. CMA-ES
   evolves the flat vector. Scales near-linearly: ~1 extra grown industrial
   zone per ~100 extra actions.
2. **The deterministic engine patch** — `randomlySeedRandom()` was reseeding
   from `gettimeofday()` after our `seed=42`. Replay variance went from
   13.6 / 173.6 / 13.6 / 13.6 → 5/5 identical. See `engine/src/initialize.cpp:78`.
3. **`varied` fitness mode** (= cityPop + built tiles + powered zones +
   distinct-tile-categories) breaks the zero-init argmax degeneracy enough
   that closed-loop policies start placing diverse stuff.
4. **`--workers 8` multiprocessing** — ~4× wall-time speedup on the M4
   (10 cores: 4 P + 6 E). Bit-identical to sequential at same seed.

## What doesn't work

1. **Pure ConvPolicy / MLPPolicy / DeepConvPolicy from empty maps.** Their
   zero-init logits give a degenerate argmax — every action lands on the
   same coordinate until CMA-ES noise breaks the tie. By then the budget
   is mostly spent. Even with the denser `growth` fitness, `deepconv`
   plateaued at obj_max=50.7 after 3 gens.
2. **ContextualTape** (tape + shared linear map-conditioning). Same
   peak as plain tape, but the `W @ features` term mostly adds noise to
   an already-optimal sequence. The extra 30 params don't help.
3. **HybridPolicy** (tape prefix + DeepConv tail) matches tape@200's
   cityPop=640 with broader QD coverage, but **doesn't exceed** it. The
   net half doesn't appear to be productively augmenting the tape's
   built city.
4. **`viewer.py` (Tk)** — silently fails on macOS Aqua Tk 8.5; canvas
   and labels render invisibly. Use `record.py` (GIF via Pillow) instead.
5. **`pop` fitness mode alone is too sparse**. With cityPop only changing
   in jumps of 160 (per industrial zone grown), CMA-ES has near-zero
   gradient until a policy stumbles into a working build.

## Engine quirks worth knowing

- **Map is 120 × 100, compile-time fixed.** All obs/policy code assumes it.
- **Power propagates through contiguous CONDBIT tiles only.** A 1-tile gap
  between a plant and a wire kills propagation. Most random placements
  never form a contiguous power network.
- **Zone growth requires `landValueMap > 0` near the zone**, computed from
  proximity to trees/water/existing development. After `clearMap`, land
  value is 0 everywhere — bootstrap deadlock unless something seeds it.
  In practice tape's "carpet-bombing" works because it places enough
  things that some end up adjacent.
- **`zone center + adjacent power conductor + adjacent road` is the
  minimal viable city**. We hand-built one and grew `cityPop=160`. The
  engine works; the search just has to find these conditions.
- Determinism is preserved only because we patched the engine and
  construct a fresh `MicropolisEnv` per evaluation.

## File map for collaborators

- `slimcity.py` — Python wrapper over the SWIG'd C++ engine.
- `policy.py` — all policy classes + `make_policy()` factory + registry.
- `evaluate.py` — `evaluate(theta, ...)` runs one or N averaged episodes
  and returns `(fitness, measures, stats_final)`.
- `qd_train.py` — CMA-ME training loop using pyribs. Has every CLI knob.
- `record.py` — renders a policy rollout as an animated GIF.
- `plot_archive_png.py` — heatmap of an `.npz` archive.
- `engine/src/initialize.cpp` — engine patched for determinism.

## Suggested next experiments

In rough order of "most likely to teach us something":

- **Random-prefix + multi-eval** (`--policy randprefix --n-evals 5`).
  Force the net to be robust across varied scaffolds; smooth the
  fitness signal so CMA-ES sees gradient further from peaks. Currently
  running in the overnight batch.
- **`growth` fitness on a known-working policy** — does it hurt tape's
  best? Currently testing.
- **Bigger archive (40×40)** to spread elites across more measure-space
  cells. May help CMA-ME's diversity-driven restart logic.
- **Wider DeepConv with a hand-coded init** that breaks zero-degeneracy
  (e.g. random N(0, 0.1) init in the first layer). Untested.
- **Different QD measures** — current (road_frac, ind_share) keeps all
  tape elites pinned to one column. Try (R, C, I) shares, or
  (n_powered, n_grown).
- **Bigger tape, longer rollouts**: `n_actions=600/1000`,
  `ticks_per_action=200`. The 1 zone per 100 actions trend should
  continue if there's no representational ceiling.

## How to reproduce a run

All saved archives carry their CLI config as metadata. To reproduce:

```python
import numpy as np
d = np.load('results/<name>.npz', allow_pickle=True)
print('policy:', str(d['policy']))
print('kwargs:', str(d['policy_kwargs']))
print('fitness:', str(d['fitness']))
print('n_actions:', int(d['n_actions']))
print('grid_dims:', d['grid_dims'])
```

Then `python3 qd_train.py --policy <name> --policy-* ... --fitness ... ...`
with those args. See `EXPERIMENTS.md` for a chronological log of what was
actually run.
