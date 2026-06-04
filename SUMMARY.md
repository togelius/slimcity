# Cross-cutting summary of all approaches

A side-by-side look at every search method we've tried on Micropolis.
For *what each one does technically*, see `RESULTS.md`. For *the
chronological narrative of how we got here*, see `EXPERIMENTS.md`.
For the layout deep-dive, see `LAYOUTS.md`.

## The seven approach families

| family | what evolves | typical params | best replay cityPop | typical wall time |
|---|---|---:|---:|---:|
| **Layout evolution** | The city itself (categorical 30×25 grid + tax) | 3,751 | **1,680** (50k evals: 1,820) | **11 min** |
| **ELM** (code-genome) | **Python source code** for `act(obs, state)` (a closed-loop policy), mutated/crossed by Claude | N/A (tokens) | **1,792** | hundreds of LLM calls |
| **Open-loop ActionTape** | A fixed sequence of `(tool, x, y)` triples | 300–3,000 | 1,120 (tape@600) | ~1 h |
| **Hybrid** (tape + net) | Tape prefix + closed-loop net tail | 5k–13k | 740 (rich_hybrid@500) | ~10 h |
| **Closed-loop ConvPolicy / DeepConv / MLP** | A single network from obs → action | 0.7k–155k | **0** | hours, scaled with depth |
| **RandomPrefix** | Stochastic random scaffold + closed-loop net | ~11k | 360 (rich_randprefix) | hours × `n_evals` |
| **ContextualTape** | Tape + shared linear "delta from features" | 330 | 0 (bonus only) | ~1 h |

## What does best — a clear ranking

Replay-verified `cityPop`, deterministic engine:

| rank | approach | cityPop | R/C/I | params | wall (8w on M4) | notes |
|---:|---|---:|---|---:|---:|---|
| 1 | **`layout` res_ind 50k evals** | **1,820** | 78/1/0 | 3,751 | ~37 min | replay #3 of stored top-3 |
| 2 | **ELM (Claude as mutator, 148 iters)** | **1,792** | (heavy R) | N/A — Python source | hundreds of API calls | **closed-loop policy in code form** — beats every CMA-ES-trained net by 5× |
| 3 | `layout` res_ind 10k evals | 1,680 | 53/2/0 | 3,751 | 11 min | first layout milestone |
| 4 | `tape` @600 varied | 1,120 | 8/0/5 | 1,800 | overnight | first R+I mix, policy-based |
| 5 | `layout` density | 1,180 | 59/0/0 | 3,751 | 33 min | archive collapsed |
| 6 | `tape` @400 varied | 800 | 0/0/5 | 1,200 | 15 min | pure industrial |
| 7 | `tape` @800 res_ind n=3 | 780 | 11/0/4 | 2,400 | ~3 h | most res, smaller total |
| 8 | **`rich_hybrid` t200@500 growth ec** | **740** | **4/0/4** | 12,752 | 9.9 h | **first closed-loop mixed R+I** |
| 9 | `tape` @300 growth ec | 660 | 1/0/4 | 900 | 22 min | first tape@300 with R |
| 10 | `tape` @200 varied | 640 | 0/0/4 | 600 | 9 min | |
| ~ | `hybrid` / `rich_hybrid` 100@300 | 640 | 0/0/4 | 5k–13k | 6 h | hybrid ceiling |
| ~ | `rich_randprefix` 50@200 n=5 | 360 | 9/0/1 | 12,452 | 5.5 h | first net-only mixed R+I |
| ~ | `randprefix` n=5 | 160 | 0/0/1 | 11k | 2.3 h | first net-based growth, single zone |
| ~ | `randprefix` n=10 | 0 | 0/0/0 | 11k | 4.5 h | over-averaging killed lucky-seed |
| ~ | every closed-loop net without scaffolding | **0** | 0/0/0 | 0.7k–155k | hours | bonuses only |

**Headline**: two very different things both reach cityPop ≈ 1,800:

- **Layout-CMA-ES** (1,820) bypasses the action-by-action problem entirely by encoding the city directly.
- **ELM** (1,792) confronts the action-by-action problem head-on — same closed-loop cadence as our failing nets — but uses Claude as the variation operator on **Python source code**. The LLM's prior on what good code looks like turns out to outperform CMA-ES on the same task that produced cityPop=0 for every CMA-ES-trained net.

Everything that uses CMA-ES on a parametric policy tops out around 1,120 (tape@600). The dramatic gap between ELM and the CMA-ES nets — both are closed-loop, both decide one tile at a time — strongly suggests **the bottleneck for the CMA-ES nets was the search algorithm + parameterization, not the closed-loop framing itself**.

## Archive coverage — different story

QD coverage is about **filling cells**, not winning the peak. Highest cell-counts:

| approach | filled / total | notes |
|---|---:|---|
| `rich_hybrid` t100@300 | 483 / 1,600 | broadest yet |
| `rich_deepconv` (8,16) growth 80g res_ind | 428 / 1,600 | rich obs broadens nicely |
| `rich_deepconv` (16,32) growth 80g | 377 / 1,600 | |
| `rich_deepconv` (16,32) growth 80g ec | 377 / 1,600 | entropy_count helps spread |
| `deepconv` (16,32,16) growth 150g | 358 / 1,600 | bonuses-only diversity |
| `rich_deepconv` (8,16) growth 120g | 356 / 1,600 | |
| `rich_deepconv` (16,32) growth 60g | 353 / 1,600 | |
| `rich_deepconv` (16,32) **pop** 80g ec | **306, all fitness=0** | clearest illustration of why dense fitness matters |
| `deepconv_wide` (32,64) growth | 54 / 1,600 | overparameterized → narrower |
| `tape@600` road_ind 40×40 | **33 / 1,600** | tape's narrow-niche pattern |
| `tape@600` res_ind 40×40 | 196 / 1,600 | same policy, richer measures |
| **`layout` res_ind** (best fitness) | **19 / 400** | every elite in the same basin |

**The two metrics move in opposite directions**: methods that win on
peak fitness fill few cells; methods that fill many cells fail to grow
anything. Tape is the high-fitness specialist; deepconv is the
"diverse failure" generalist; hybrid sits between (high fitness on the
left edge + diversity from the net half elsewhere).

## How does the choice of descriptor affect things?

We tried four QD measure pairs:

| measures | what they encode | observed effect |
|---|---|---|
| `road_ind` (`road_frac`, `ind_share`) | original — fraction of build that's road, fraction of zoning that's industrial | Pins tape elites to the `road_frac=0` column (tape places no roads). Layout fills the high-`ind_share` row. Wastes ~60% of the archive. |
| `res_ind` (`res_share`, `ind_share`) | R-share vs I-share within zoned tiles | Best for most policies. Spreads tape@600's elites 6× wider (196 vs 33 cells). Layouts fill the (0.3, 0.3) niche but the rest is unreachable. ~50% of archive is mathematically dead space (`res + ind ≤ 1`). |
| `density` (`built_density`, `infra_density`) | how full the map is + how much is infra | **Killed the layout archive** (only 2/400 cells filled) because the wire grid is excluded from `is_road` (the engine's wires are 208–222, not the 64–206 ROAD_RANGE). Also killed randprefix exploration (1/1600 cells). Wrong descriptor for layouts; OK for randprefix-style policies that build roads, not great for any. |
| `entropy_count` (Shannon entropy of category counts × normalized build count) | "diverse, dense" curriculum | **Helped tape@300 grow R+I where it previously couldn't.** Spreads closed-loop nets broadly (~300 cells). Acts as a curriculum: bottom-left = "did nothing"; bottom-right = "spammed one thing"; top-right = "diverse and dense → where cities live". Notice the `rich_deepconv + pop fitness` heatmap: 306 cells filled, all at fitness 0 — the curriculum spreads policies even when fitness gives no gradient. |

**Pattern**: descriptors that capture **what's on the map** spread better than descriptors that capture **what kind of city it is**. Tape is so narrow in city-shape that any "city kind" descriptor pins it; descriptors based on tile counts give it room to vary.

## How does the choice of fitness affect things?

| fitness | what it gives credit for | observed effect |
|---|---|---|
| `pop` | `cityPop` delta only | Too sparse. cityPop only changes in jumps of 160 (per grown industrial zone). The 1000-eval baseline with `conv + pop` had every elite at fitness 0. **The clearest data point**: `rich_deepconv + pop + entropy_count` filled 306 cells, every single one at fitness exactly 0. |
| `dense` | + 0.01 × built tiles + 1 × powered zones | Unblocked the closed-loop policies from "every action at (0,0)" degeneracy. cityPop still 0 for nets, but now they're at least placing things. |
| `varied` | + 2 × (# distinct tile categories ever placed) | Best for `tape` — gave the carpet-bomb strategy its variety bonus. Saturates at +12 once everything has been placed once. |
| `growth` | + 0.5 × tiles past ungrown stamp + 0.1 × (zone, neighbor-road) pairs | Needed for hybrid / randprefix. Gives intermediate signal *between* fitness jumps and rewards spatial connection. Without it, the net half can't see partial progress toward viable cities. |

**Multi-eval averaging** (`--n-evals N`) interacts here:
- n=5 was the sweet spot for `randprefix` (cityPop=160, the only growth)
- n=10 was *worse* than n=5 (all 0): heavier averaging selects for robustness over peak performance on rare-success landscapes
- n=3 in `tape@800 res_ind` got cityPop=780; without n=3 the same theta would replay noisily

## Param count vs cityPop

Plotting params (log scale) vs best replay cityPop, three regimes:

```
2000 │                                       layout/ELM ★
1800 │                                          ★
     │
1600 │
1400 │
1200 │
1000 │     tape@600 ●
 800 │   tape@400 ●           rich_hybrid t200 ◆
 600 │ tape@200 ●        hybrid ●  rich_hybrid t100 ◆
 400 │                                rich_randprefix ◆
 200 │tape@100●        randprefix n=5 ●
   0 │all closed-loop nets ──────────  rich_deepconv ●  mlp_wide ●
     └─────────────────────────────────────────────────────────────
     300    1k     3k     10k    30k    100k         params
```

- **<3,000 params**: tape is the queen of this regime. ActionTape just needs enough triples; adding sophistication hurts.
- **3,000–4,000 params**: layout / ELM thrives. The categorical-grid encoding fits perfectly into this range.
- **5k–13k params**: hybrid composites earn their keep — they don't beat tape but they show diversity and the rich variants started producing real (closed-loop-net-influenced) cities.
- **>30k params (pure nets)**: hopeless in our eval budget. CMA-ES can't navigate the space with 5k–15k samples; we got bonus-gaming, not cities.

## What works (final pattern)

1. **Two paths to the top — both bypass CMA-ES + dense vectors.** Layout-CMA-ES wins by encoding the artifact, not the process. ELM wins by keeping the closed-loop framing but using an LLM-mutator over **code** instead of CMA-ES over a dense parameter vector. Tape (pure CMA-ES on a dense vector) is third, ~30% behind both.
2. **Scaffolding beats from-scratch (for CMA-ES policies).** Tape works because it carpet-bombs; randprefix works (a little) because random placements happen to be a scaffold; hybrid matches tape because tape provides scaffolding for the net half.
3. **Rich obs gives nets visibility, not victory.** They went from cityPop=0 → cityPop=360–740 with rich obs, but still below scaffold-free open-loop tape. The structural problem (one-tile-at-a-time + sparse rewards + argmax decode under CMA-ES) survives every perceptual upgrade — until you swap CMA-ES + nets out for an LLM-mutator + code, at which point closed-loop suddenly works (ELM 1,792).
4. **Curriculum descriptors push search toward viable regimes.** `entropy_count` got tape@300 to find R+I where it previously couldn't.
5. **Dense fitness is mandatory for any closed-loop CMA-ES search.** `pop` alone produces 0 for every parametric net.

## What doesn't work

1. **Closed-loop ConvPolicy / DeepConv / MLP from scratch on empty maps.** Across 1.5 weeks and many configurations, *none* produced cityPop > 0 without a scaffold or rich obs.
2. **Heavier multi-eval averaging.** n=10 was worse than n=5 — it dilutes lucky-seed strategies on rare-success landscapes.
3. **`density` descriptor on layouts.** Wire grids aren't counted in `is_road`, so all layouts collapse to the same `infra_density=0`.
4. **Pop-only fitness for non-tape policies.** Network argmax is degenerate from zero-init; without intermediate signal the search never escapes.

## Engine quirks that shaped everything

- **Wires conduct power; plain roads don't.** Wire tiles 208–222 have `CONDBIT`; roads 64–206 do not. The layout work depends on this — a wire grid both conducts power and reaches every cell.
- **`generateSomeCity → clearMap` deadlocks growth.** Land value is computed from proximity to trees/water/existing development; after `clearMap`, land value is 0 everywhere and zones can't bootstrap. Tape works because carpet-bombing seeds enough things adjacent to each other.
- **Determinism is fragile.** We patched two bugs (`randomlySeedRandom` on reset, `_mapData` uninitialized in Map<>), but there's a residual ~5–10% replay variance past ~52k simTicks that we couldn't track down. Layouts have a particularly visible replay gap (~700 fitness between stored and replay). `--n-evals 3` or `5` is the workaround.

## File map (for collaborators)

- `slimcity.py` — Python wrapper for the SWIG'd C++ engine
- `policy.py` — all parametric policy classes + `make_policy()` factory + registry
- `evaluate.py` — `evaluate(theta, ...)` runs one or N averaged episodes
- `qd_train.py` — CMA-ME loop (also supports `--init-archive` warm-start, used heavily by the layout work)
- `elm/` — **ELM (code-genome MAP-Elites)**: separate harness where the genome is Python source for `act(obs, state)` and the variation operator is Claude. `elm/elm_train.py` is the driver; `elm/sandbox.py` defines the restricted execution environment and primitives; `elm/operator.py` has `LLMOperator` + `MockOperator`; `elm/seeds.py` has hand-written starter policies; `elm/archive.py` mirrors the QD archive on the CMA-ES side. See `elm/README.md`.
- `record.py` — render a policy rollout as an animated GIF
- `plot_archive_png.py` — heatmap of an `.npz` archive (auto-detects measures)
- `perturb_study.py` — neighborhood analysis around the best elite of an archive
- `analyze_layout.py` — post-hoc layout characterization (tax dist, category shares, co-occurrence)
- `engine/src/` — patched C++ engine (determinism + zero-init bugs)
- `RESULTS.md` — what works / what doesn't / leaderboard
- `EXPERIMENTS.md` — chronological log of every batch
- `LAYOUTS.md` — deep-dive on layout-evolution findings (what Micropolis rewards)
- `AGENTS.md` — short note to LLM agents working on this repo
