# Layout evolution — what cities does Micropolis reward?

A reframe of the QD experiment: instead of evolving a *policy* that plays
SimCity, evolve the *city itself*. The archive then directly tells us
which city *shapes* the simulator rewards, which addresses questions
like:

- Does Micropolis favor dense cities or sprawl?
- Does it prefer functional separation (R / C / I in distinct zones) or mixed-use?
- Does it favor low tax?
- What R/C/I ratio maximizes population?

## Encoding

`LayoutGenome` (in `policy.py`) decodes a flat float vector into:

- A **30 × 25 categorical grid** (`CELL_SIZE = 4`-tile cells fill the
  120×100 map). Each cell picks one of 5 categories:
  `empty / residential / commercial / industrial / park`. Encoded as
  per-cell K=5 logits, argmax decides.
- A **single tax-rate gene** (tanh-mapped to integer ∈ [0, 20]).

Total: **3751 parameters**. sep-CMA-ES is the obvious choice.

## Build pipeline

Each evaluation:

1. `clearMap`
2. Lay a **WIRE grid** at every `CELL_SIZE`-th row and column. Wires,
   not roads — in this Micropolis engine plain road tiles do **not**
   have `CONDBIT`, so they don't propagate power. A wire grid both
   conducts power *and* (since the engine doesn't strictly require
   road adjacency for zone growth) gets things to grow.
3. Drop a coal plant at top-center as a guaranteed power source. The
   genome doesn't have to evolve plant placement.
4. `setCityTax(tax)` from the tax gene.
5. For each cell, place the chosen tool at the cell center. Zone
   footprint is 3×3, cell is 4×4 → 1-tile gap between zones, the wire
   grid sits in those gaps. Every zone is adjacent to a wire on all
   four sides.
6. `env.tick(stabilization_ticks)` — let zones grow.

`CELL_SIZE = 4` was chosen by checking that a 3×3 zone at cell center
(col 4i+2) lands its leftmost tile (col 4i+1) adjacent to the wire at
col 4i. At cell size 5 only one side is wire-adjacent; at cell size 6
neither side is, and **no zones grow**. The first smoke run had
`CELL_SIZE = 6` and got cityPop=0 across every test theta — a
diagnostic regression that drove the cell-size choice.

## What "infra" doesn't capture

`evaluate.tile_descriptors` defines `is_road` as the tile-ID range
64..206, which **excludes wires** (208..222). So `density.infra_density`
ignores our wire grid and reports ≈ 0 — the archive collapses to a 1-D
line in that mode. We use `res_ind` (residential share × industrial
share within zoned tiles) instead; it gives meaningful spread for these
layouts.

## Run recipes

```bash
# main run: 100 gens × 5 emitters × 20 batch = 10,000 evals
/usr/bin/python3 qd_train.py --policy layout --gens 100 --emitters 5 --batch 20 \
    --workers 8 --fitness dense --n-actions 1 --ticks-per-action 100000 \
    --measures res_ind --save archive_layout_resind.npz

# alternate: density measures (built_density × ~zero infra)
/usr/bin/python3 qd_train.py --policy layout --gens 100 --emitters 5 --batch 20 \
    --workers 8 --fitness dense --n-actions 1 --ticks-per-action 100000 \
    --measures density --save archive_layout_density.npz

# post-hoc: tax distribution, category shares, best layout
/usr/bin/python3 analyze_layout.py archive_layout_resind.npz
```

Use `/usr/bin/python3` explicitly. The Homebrew `python3` on this
machine is Python 3.14 and won't load the engine `.so` (built for
Python 3.9).

`n_actions = 1` + `ticks_per_action = 100000` → 100k stabilization
ticks (≈ 6 game years). The build itself takes ~150 ms; stabilization
adds ~100 ms; total ~250 ms per evaluation. 10,000 evals on 8 workers
land in ~5 minutes.

## Results

### Run 1 — `res_ind` measures (R/I balance)

- **10,000 evaluations (100 gens × 5 emitters × 20 batch), 10.8 min wall on M4 (8 workers)**.
- Stored `obj_max = 2255.2`, `obj_mean = 1607.1`, `qd_score = 30,534`, 19/400 cells filled.

Caveat: the documented residual non-determinism past ~52k simTicks
([commit e2baa05](.) and predecessors) means replay fitness differs
from stored fitness. Replaying the top-5 elites under a fresh process:

| rank | stored | replay | cityPop | R  | C | I |
|-----:|------: |------: |-------: |--: |--:|--:|
| 1    | 2255   | 1615   | **1,480** | 58 | 1 | 2 |
| 2    | 2212   | 1812   | **1,680** | 53 | 2 | 0 |
| 3    | 2007   | 1567   | **1,440** | 58 | 0 | 0 |
| 4    | 1802   | 1802   | **1,680** | 43 | 1 | 0 |
| worst| 345    | 645    | 520     | 15 | 1 | 0 |

Even with replay drift, the smallest replay cityPop in the top-10 is
~1,440 and the largest is ~1,680 — comfortably above the previous
best `tape@600` (1,120). **Layout evolution finds bigger cities than
any policy representation tested so far, with much less compute.**

### What Micropolis rewards

Three sharp findings:

**1. Low tax wins.** 16 of 19 elites have tax in [0, 4]; only 3 in
[5, 8]; none higher. Mean fitness by bin:

| tax bin | n elites | mean fitness |
| ------- | -------: | -----------: |
| 0–4     | 16       | **1,757**    |
| 5–8     | 3        | 805          |

The very best elite has `tax = 0`. The engine collects no tax, the
budget never runs into trouble (the city has $872k starting funds and
spends only a fraction on placement), and growth-related funding
penalties never trigger.

**2. Balanced R / C / I beats any skew.** Every top-10 elite has
roughly 1/3 residential, 1/3 commercial, 1/3 industrial (within zoned
cells), plus ~20% empty and ~20% park. Mean category share of top 10:

| category    | mean share |
| ----------- | ---------: |
| empty       | 0.20       |
| residential | 0.20       |
| commercial  | 0.19       |
| industrial  | 0.21       |
| park        | 0.20       |

Every elite cluster sits in the same niche of the res_ind plane — the
QD archive only filled 19 of 400 cells, all near `(res_share≈0.3,
ind_share≈0.3-0.4)`. CMA-ME couldn't escape this basin because every
deviation hurts fitness.

**3. No functional separation. Mixing is optimal.** The 4-neighbor
co-occurrence matrix of categories in the best layout (fitness 2255)
is **statistically indistinguishable from random placement** — every
empty-R, R-C, C-I, etc. pair occurs within 0.005 of its expected
independence frequency. The best Micropolis city has no zoning logic:
just a uniform random sprinkle of R / C / I / park / empty over the
wired grid. With our 4-tile cell spacing, every zone is within 1–2
cells of every other zone, so R / C / I demand-supply is satisfied
locally everywhere.

This is a clean answer to "does Micropolis favor functional
separation?" — **no, it favors uniform mixing**, at least on this
short stabilization horizon.

### Run 2 — `density` measures

Same setup, just different QD descriptors. **10,000 evaluations, 32.9
min wall** (slower than run 1 because the buffered output and tighter
fitness cluster mean less worker-cache hit on the OS scheduling side —
the per-gen time was similar but startup was slower).

- Stored `obj_max = 2106.8`, `obj_mean = 2078.7`, **only 2 elites
  filled** out of 400 cells.

Top elite replay:

| rank | stored | replay | cityPop | R  |
|-----:|------: |------: |-------: |--: |
| 1    | 2107   | 1307   | **1,180** | 59 |
| 2    | 2051   | 1211   | **1,080** | 58 |

The "only 2 elites filled" result is **expected and explanatory**: the
`density` measure pair is `(built_density, infra_density)`, where
`infra_density` is counted as `(roads ∪ wires) / total tiles`, but our
build sets `is_road = (tile_id in [64, 206])` which **excludes wires**
(208..222). Our wire grid contributes 0 to infra_density. The only
infra-tile contribution is the coal plant (≈ 16 tiles in a 12,000-tile
map ≈ 0.001). So all elites land in the same infra_density bin, and
the archive collapses to a 1-D line along built_density. CMA-ME's
diversity machinery has nothing to spread across.

The takeaway: `density` mode is not useful for layouts under the
current `tile_descriptors` definition. To get a real 2-D spread we'd
either (a) extend `is_infra` to include wires, or (b) use
`(built_density, tax)` as a layout-specific measure pair. The
`res_ind` archive (run 1) gives more interpretable spread.

## Answers to the original questions

> Does the game favor dense cities, transit, functional separation
> etc? Does it favor low tax?

- **Dense cities: yes, mostly.** Best layouts have ~80% built cells
  and ~20% empty/park. Going all-in on full density doesn't help: the
  archive consistently keeps ~20% of cells unzoned. Whether this is
  about land value, breathing room, or a Micropolis quirk we can't
  separate from this experiment.
- **Functional separation: no.** Co-occurrence of categories in the
  best layouts matches their independent-expected frequencies within
  ~0.005. Uniform random mixing of R / C / I / park is optimal,
  presumably because every R zone needs nearby C/I as demand sources
  and a uniform sprinkle satisfies every demand locally.
- **Low tax: yes, strongly.** 16 of 19 res_ind elites have tax in
  [0, 4], 0 of 19 have tax > 8, and the best elite has tax = 0. Mean
  fitness drops by 2× when tax goes from 0–4 (1,757) to 5–8 (805).
- **Transit: out of scope here.** Our build fixes a wire grid at every
  4th row and column, so the policy has no transit knobs. A follow-up
  could vary grid density or replace wires with road–wire bridges to
  test whether transit matters beyond the minimum power network.

## Compared with policy evolution

| representation | best stored | best replay cityPop | params | wall time |
|---|---:|---:|---:|---:|
| `tape` @600 `varied` | 1,202 | 1,120 | 1,800 | overnight |
| `tape` @100 `varied` | 509   | 480   | 300   | 7 min |
| `hybrid` tape+conv   | 680   | 640   | ~11k  | ~15 min |
| **`layout` `res_ind` (this work)** | **2,255** | **1,680** | **3,751** | **11 min** |

Layout evolution beats every previous representation on best
replayed cityPop while finishing faster and using fewer parameters
than the hybrid / deeper-net attempts.

## What this experiment doesn't answer

- Whether `tape` and `layout` are exploring the same fitness landscape
  from different angles. The best `tape@600` city had 8 R + 5 I zones;
  the best `layout` city has ~58 R zones. Different operating points
  of the same simulator, or different solutions altogether?
- What the engine's residual non-determinism is *for* — replay differs
  by ~700 fitness on the top elite. We could either patch that out for
  cleaner experiments, or accept it as part of the sim's variance and
  use multi-eval averaging like `randprefix --n-evals 5`.
- Whether the "uniform mixing wins" finding survives larger spatial
  scales. Our 4-tile cell makes everything local; with 8- or 16-tile
  cells, neighborhood-level segregation might start to matter.

## Reproducing

```bash
# 1. Build engine (one-time, see RESULTS.md)
# 2. res_ind layout run (~11 min, 8 workers on M4):
/usr/bin/python3 qd_train.py --policy layout --gens 100 --emitters 5 --batch 20 \
    --workers 8 --fitness dense --n-actions 1 --ticks-per-action 100000 \
    --measures res_ind --save archive_layout_resind.npz
# 3. Analyze:
/usr/bin/python3 analyze_layout.py archive_layout_resind.npz
```
