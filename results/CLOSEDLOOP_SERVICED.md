# Closed-loop champion v2: service-cap break (`closedloop_serviced_best.py`)

A new best closed-loop policy for building Micropolis cities, at the standard
episode shape (120 actions × 100 ticks, empty map, deterministic engine).

## Result

Replay-verified `cityPop`, robust mean over the canonical 5 rolls (seeds 42–46):

| policy | cityPop (5-roll) | 15-seed mean (42–56) | 15-seed min |
|---|---:|---:|---:|
| prior closed-loop champion (`elm_clind_long_best`) | 15,076 | 14,728 | 13,140 |
| **`closedloop_serviced_best` (this work)** | **18,324** | **18,011** | **16,360** |

**+22%, and it dominates the champion on every single seed** — the new policy's
15-seed *minimum* (16,360) exceeds the champion's 15-seed *maximum* (15,660).
Genuinely closed-loop (counterfactual-sensitivity `cf_sensitivity ≈ 0.65`,
identical to the parent); every placement after step 0 is chosen by reading the
live `obs.tile_map`.

## What changed and why

The policy extends the prior champion (a reactive residential blob-builder) with
two levers **no prior elite used — 0 of 417 archive elites place a single
stadium, police, park, airport or seaport**:

1. **One STADIUM (the decisive lever).** Reverse-engineering the engine
   (`message.cpp` case 26) shows `resCap = True` fires once `resPop > 500` and no
   stadium exists, which pins `resValve` to 0 and stalls residential
   densification at ~800 resPop. A single stadium clears the cap, so the demand
   valve stays saturated and the (already powered, road-served) residential blob
   keeps densifying to ~970 resPop. The stadium is dropped at a spine-row gap
   `(68, 46)` between clusters so the reactive builder simply routes around it.
   `stadiumPop++` is unconditional in the census, so the cap clears even though
   the build never bothers to wire the stadium.

2. **NUCLEAR instead of COAL power.** Zero pollution (vs ~56 avg) → higher land
   value → higher achievable density. Small but free, and stacks with the
   stadium.

## Why this is near the ceiling for the 120-action budget

`cityPop = (resPop + (comPop + indPop)·8)·20`. The growth chain is a cycle
R→C→I→R (each zone type must road-reach the next to densify). Findings from
~hundreds of controlled rollouts:

- **The hard constraint is the 120-action budget, not map space.** A full
  wire+road grid over any useful area needs 400–800 placements. Roads/wires do
  not conduct to each other and fragment power, so dense road-service of many
  zones is unaffordable.
- **The structure tops out at ~28 powered + road-served zone slots**, each of
  which the builder grows to ~density 3 (≈35 resPop/zone). Adding more zones
  (e.g. converting the 48 "wasted" commercial slots to residential) does not
  help — the extra zones land in unpowered/unserved positions and never grow.
  The commercial slots are not wasted: they serve as the traffic destinations
  residential needs to densify past low density.
- **Balanced R/C/I cities are worse here**, confirming the original search:
  commercial/industrial densify far too slowly within 12,000 ticks to cash in
  their ×8 multiplier, while residential bootstraps readily. Best balanced
  hand-builds reached only ~1,800 cityPop.
- **Police cut crime sharply (~110 → ~30) and raise land value**, but cost more
  action-budget than the marginal density buys, and disrupt the reactive
  builder's fragile bootstrap when injected into its core. Net negative here.

## Reproduce

```bash
PYTHONPATH=. /usr/bin/python3 - <<'PY'
from elm.evaluate_code import eval_code
import numpy as np
src = open('results/closedloop_serviced_best.py').read()
pops = [eval_code(src, seed=s, n_actions=120, ticks_per_action=100,
                  fitness_mode='dense', measures_mode='res_ind').city_pop
        for s in range(42, 47)]
print('mean cityPop', np.mean(pops), pops)
PY
```

## Action cadence — a tick is ~half a day

Engine timing (verified in `simulate.cpp` / `fileio.cpp`): **16 simTicks = 1
`cityTime` unit, and 48 `cityTime` = 1 year**, so one `simTick` ≈ half a day,
the default **100 ticks/action ≈ 1.5 in-game months**, and the canonical
12,000-tick episode ≈ **15.6 in-game years**. Acting only every ~1.5 months is
sparse — zones placed late barely get time to densify.

Holding **total game-time fixed at 12,000 ticks** and acting more often (same
policy, finer cadence) is a free win — extra cook time for the zones:

| cadence (n_actions × ticks) | champion | champion + stadium |
|---|---:|---:|
| 120 × 100 (canonical) | 14,720 | 18,106 |
| 300 × 40 | 17,344 | **19,504** |
| 600 × 20 | 16,830 | 19,504 |

The stadium lever holds at every cadence (+12–23%). The policy file is
cadence-agnostic (it just reads `obs`), so the *same* file scores 18,106 at the
canonical cadence and ~19,504 at 300×40.

It plateaus at ~19,500 because the reactive builder is **structurally capped at
~30 powered + road-served zone slots** — finer cadence only buys cook time, not
more zones. Bigger open-loop grids place hundreds of zones and reach ~20k, but
balanced R/C/I cities (needed to cash in the ×8 com/ind multiplier) never
densify within ~15 game-years, so ~20k is near the blank-map ceiling regardless
of cadence.
