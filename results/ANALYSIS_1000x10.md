# Top-policy analysis at 1000 actions × 10 ticks

Episode: **1000 actions × 10 ticks = 10,000 ticks ≈ 13 game-years**, acting every
~3 game-weeks. Empty map, deterministic engine, mean over 7–12 seeds.

## Leaderboard

| policy | kind | cityPop | powered zones | uses budget? |
|---|---|---:|---:|---|
| prior champion (`elm_clind_long_best`) | closed-loop | 16,157 | 30 | no — caps at ~120 placements |
| serviced champion (`closedloop_serviced_best`) | closed-loop | 19,477 | 30 | no |
| ELM diverse (`elm_diverse_overnight_best`) | blueprint | 2,369 | 231 | yes, but balanced→fails |
| **scalable serviced comb (`scalable_comb_1000`)** | **blueprint** | **83,046** | **223** | **yes** |

**~4.3× the best previous policy at this episode shape**, robust (12-seed mean
83,322, min 79,480, std 2.6%), 0 runtime errors through `eval_code`.

## Why the existing closed-loop policies don't benefit

Both reactive blob-builders are **structurally capped at ~30 powered + road-served
zone slots** — they build a fixed 4-cluster blob and then return `None`. Giving
them 1000 actions instead of 120 changes nothing: they place their ~120 tiles and
idle for the remaining ~880 actions. So finer cadence only buys them a little cook
time (15.9k→16.2k, 18.1k→19.5k). The ELM blueprint *does* place 231 zones but uses
a balanced R/C/I mix, which never densifies in ~13 game-years (commercial/industrial
grow far too slowly), so it stalls at 2.4k.

## What the scalable comb does differently

It is built so that **every zone is both powered and road-served**, and it **scales
with the action budget**:

- **Power**: one contiguous vertical residential *spine* (a single power network);
  11 horizontal zone *bands* hang off it, each band contiguous back to the spine, so
  one nuclear plant powers all ~220 zones (vs the champion's 30).
- **Access**: each band is just 2 zone-rows thick, sandwiched against a shared road,
  so every zone has perimeter road access (needed to grow at all).
- **Cook time**: built **band-by-band** (road + its zones together), so early bands
  densify for nearly the whole episode. Interleaving this way alone was +35%
  (60k→82k) over placing all roads first.
- **Services** (the levers no prior elite used): one **stadium** (clears `resCap`,
  which otherwise pins residential at ~500 resPop), ~18 **police** (crime 130→~20),
  **nuclear** power (zero pollution), and a sprinkle of **commercial** as the traffic
  destinations residential needs to densify (removing it halves resPop: 32k vs 80k).

## Where the ceiling is

Even at 83k the zones sit at only **~density 1** (resPop ≈ 3,400 from 223 zones ≈
15/zone; the max is 40/zone). The binding constraint is **land value ≈ 54** — below
the ~94 needed for the growth term `evalRes` to go positive, so zones grow slowly.
If every zone reached max density, cityPop would approach ~180k. The natural land-value
lever (parks) **fragments the power network** (parks don't conduct), dropping powered
zones 223→167 and net cityPop, so it's off the table for this structure. Raising land
value without breaking power is the open problem; ~83k is the practical ceiling for
this layout at this cadence.

## On "closed-loop"

The comb is a deterministic **blueprint** (an open-loop builder in the project's
taxonomy, like the ELM diverse champion — it replays a plan and ignores `obs`). The
project's own finding holds: the performance *peak* on this task is open-loop. The
best genuinely closed-loop (obs-reactive) policy at this episode shape remains the
serviced champion at ~19.5k, because a reactive builder that covers this much area is
effectively executing the same fixed layout anyway.
