# ELM run: elm_comb1000_overnight (1000 actions × 10 ticks)

Quality-Diversity (MAP-Elites) over Python-code closed-loop policies, Claude
(sonnet) as the mutation/crossover operator. Seeded with the scalable serviced
comb (`comb1000`, ~83k) plus the top closed-loop champions from prior runs and
the diverse hand-written founders. Behavior = `cl_ind` (closed-loopness:
cf-sensitivity, trajectory-divergence; + industrial share), fitness = dense
(cityPop-shaped), 3 rolls/eval, 8×8×10 archive.

## Headline
- **Best city: cityPop 127,340** (single roll; robust mean 120,640 over 6 seeds,
  min 110k), found at iteration 8,494 via `mutate+cl`. Genuinely closed-loop
  (cf-sensitivity 0.918).
- **Seed → champion: 82,767 → 127,340 (+54%).** The operator improved a strong
  seed, it did not start from scratch.
- Ran **~49 h / 16,392 iterations**; archive **87.2% filled** (558 / 640 cells).
- Origins of elites: mutate+cl 185, mutate 173, crossover 108, crossover+cl 90.

## What the operator discovered (on top of the seeded comb)
The seed was a spine+bands comb with one stadium. Claude evolved it into a
full-service city:
- **Airport** (pop>55k) — clears `comCap` (comPop>100), so commercial grew to
  ~280 and now contributes ~34% of cityPop via the ×8 multiplier.
- **Seaport**, **2nd stadium**, **2nd & 3rd nuclear plants** (pop/step-triggered,
  to power the larger city).
- **Fire stations** (early + reactive extras) — a lever never explored by hand:
  at high density fires break out and destroy zones; fire coverage protects them.
- **Scaled police** (early + every ~100 steps) for crime.
- **Adaptive zone mix**: `pick_zone_type` reads the live R/C composition vs
  pop-dependent targets — genuinely reactive — plus reactive wire-repair toward
  unpowered zones.

## Interpretation (the archive map)
Binning elites by closed-loopness × industrial share, colored by cityPop, shows
one clean gradient: **every high-pop city sits in the zero-industry row; anything
with meaningful industry collapses to ~1k; and the global best is the most
closed-loop, zero-industry cell.** Industry's ×8 multiplier never pays off in
~13 game-years (it densifies far too slowly), while residential + a thin
commercial layer (for the R→C traffic that drives density) scales cleanly.

## Progress shape
Fast early jump (83k → 101k by iter 83 as services were added) → plateau →
105k (~iter 4,170) → the big step to 127k (iter 8,508, the full-service champion)
→ flat for the last ~8,000 iters. The long plateau suggests ~127k is a strong
local optimum for this structure; the remaining limiter is land-value/density
(resPop 4,329 over ~220 zones ≈ 20/zone vs a per-zone max of 40).

## Artifacts
`results/elm_comb1000_overnight.npz` (archive + every cell-winner's source),
`results/elm_comb1000_overnight_best.py` (champion), `..._SUMMARY.md` (this file).
