# The best city-building policy (hand-designed): `megacity`

Goal: the highest-population city this engine can grow. Built by combining
classic SimCity/Micropolis strategy with an empirical search *on this engine*
(the citylab sweeps), then distilled into one open-loop policy.

## Result

**Sustained cityPop ≈ 14,400** (5-seed mean), **trough 5,100, peak 26,720** —
versus the previous best of **4,460** (evolved ELM) and **1,820/2,120** (layout
CMA-ES). Even the *worst trough* beats the old record.

![megacity](docs/megacity.png)

Dense residential columns (green) fed by wire spines (grey) with nuclear plants
(black) distributed along them. Policy source: `results/best_policy_megacity.py`
(registered as the `megacity` seed in `elm/seeds.py`).

## What the search found about *this* engine

Strategy guides say: balance R/C/I, keep tax low, supply power and roads, manage
pollution and land value. Tested on the engine, the decisive levers were:

1. **Power supply is THE bottleneck.** Zones conduct power to their neighbours,
   so a block fed by one spine partly lights up — but each plant has a *finite
   supply*. Going from a few plants to ~12 distributed plants took the same
   block from ~4k to **>20k** population. Powering every zone is the whole game.
2. **Density beats coverage.** Pack 3×3 zones **edge-to-edge** (3-tile pitch),
   not on the gappy 4×4 grid the layout-CMA-ES used. A **wire spine every 4th
   column** injects/distributes power (wires conduct, roads don't) at low cost.
3. **Pure residential maximises population.** Commercial barely helps;
   **industrial craters it** (pollution destroys land value → growth collapses).
   This contradicts the generic "balance R/C/I" advice — on this short-horizon
   engine, residential density dominates.
4. **Nuclear > coal on mean population.** Nuclear has higher supply and low
   pollution. Coal is *steadier* (no meltdowns) but lower; nuclear's mean and
   peak are higher despite occasional meltdown busts. (Coal is the
   lower-variance alternative if you want stability over peak.)
5. **Cities oscillate (boom-bust).** Dense Micropolis cities swing ~5×, so the
   honest metric is **sustained population** (averaged over seeds × late
   time-points), not a lucky single-tick peak.

## The policy

Open-loop blueprint (`act` precomputes the plan, replays one tile/step, then
idles to stabilise): one 36×30 residential block on a 3-tile pitch, a wire spine
every 4th column, 12 nuclear plants distributed on the spines. ~1,632
placements. See `elm/seed_megacity.py` for the ~20-line generator.

## Caveats / budget

- Needs a large action budget (`n_actions ≳ 1,650`) and long stabilisation
  (~100–150k ticks) — far more than the 120-action / 12k-tick budget the evolved
  records used, so it is **not** an apples-to-apples comparison to 4,460; it's a
  bigger city under a bigger build budget.
- It is **open-loop** (ignores `obs`); it would be discarded by the
  `--min-reactivity` closed-loop filter. It's a "best artifact", not a reactive
  controller.
- cityPop oscillates; any single rollout lands somewhere in ~5k–27k.

## Reproduce

```python
from elm.seeds import SEEDS; from elm.sandbox import compile_genome
from slimcity import MicropolisEnv
act,_ = compile_genome(SEEDS["megacity"])
env = MicropolisEnv(seed=42); s={}
while True:
    a = act(type("O",(),{})(), s)
    if a is None: break
    env.place(*a)
env.tick(150000); print(env.stats.city_pop)
```
