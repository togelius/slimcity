# Tiling the champion + starting from terrain

Two follow-ups to the ELM `clind_long` champion (cityPop ~15,380), which packs
dense residential into a single ~16-row band and leaves the rest of the map
empty. Both are **hand-constructed** builds (`tile_and_terrain.py`), not evolved
— they test hypotheses the evolved runs suggested.

## 1. Tile the dense pattern across the whole map

The champion's motif = dense 3×3 zones, each touching a power wire and a road,
fed by a connected wire grid (vertical wires + a horizontal bus) with power
plants on the bus and horizontal roads for traffic. Tiling that over the full
120×100 map (cell=4 → ~696 zone slots):

| zone mix | peak cityPop | R / C / I | note |
|---|---:|---|---|
| pure R | ~26–30k | 1188 / 0 / 0 | residential demand alone scales surprisingly far |
| **R/R/R/C** | **~40,620** (peak 41,600) | ~1300 / ~120 / 0 | **best — commercial supplies the jobs that let R grow** |
| R/R/C/I | ~3,200 | 0 / 0 / ~10 | **collapses** — industrial pollution tanks land value, R dies |
| R/C/I | ~2,700 | ~0 / 0 / ~15 | collapses |

**Result: tiling more than 2.5×'s the champion — ~40k vs 15k.** The champion
was leaving ~84% of the map idle. Two clear lessons:

- **Commercial is the multiplier.** Pure-R caps around 26–30k (residential
  demand saturates without jobs); adding ~25% commercial feeds that demand and
  pushes to ~40k. Industrial *looks* like jobs too but its pollution destroys
  land value and the whole thing collapses to near-zero.
- **The bottleneck was never the optimizer's pattern — it was coverage.** A
  good local recipe tiled globally beats a great local recipe used once.

Render: `docs/champions/tiled_rrrc_40k.{png,gif}`.

## 2. Start from a water + forest map

`generateSomeCity` (no `clearMap`) yields a natural map: ~4,832 water + ~2,922
tree tiles. Water/trees feed the engine's land-value term, so land value rises
to **57–64** vs **~36** on bare cleared land. A water-aware version of the same
grid (routes around water, plants on land):

| start map | peak cityPop | land value |
|---|---:|---:|
| cleared (bare) | ~40,620 | 36 |
| **water + forest** | **~32–34k** | **57** |

**Result: terrain did NOT beat cleared — slightly worse.** Higher land value
per zone (57 vs 36) is real, but water occupies ~40% of the map, so far fewer
zones fit. The lost buildable area outweighs the density bonus for a naive
grid. Render: `docs/champions/terrain_rrrc_32k.png`.

The honest read: a *rigid grid* can't exploit terrain — it just loses area to
the water. Terrain should help a layout that **hugs the shoreline** (max
land-value contact, minimal water wasted), which is exactly the structure a
QD/ELM search could discover if pointed at terrain maps. That's the natural
next experiment: run layout-CMA-ES or ELM on generated terrain instead of
`clearMap`.

## Caveats

- Hand-constructed, not evolved — these are existence proofs, not search results.
- Replay variance: peak cityPop wobbles ~±10% run-to-run (the documented
  post-52k-tick non-determinism); pure-R read 26.5k and 30k on two runs, RRRC
  36k–41.6k. "peak" is the max over a 400k-tick rollout; cities boom then
  partially decline, so final < peak.
- Reproduce: `/usr/bin/python3 tile_and_terrain.py`.
