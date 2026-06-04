# slimcity

Quality-Diversity (CMA-ME) experiments over a headless Micropolis (SimCity 1)
engine. Population-based search where each elite is a tiny convolutional
policy that places tiles into a 120×100 map, fitness is the resulting in-game
population, and a 2-D behavior space spreads elites across road-density and
industrial-share axes.

The C++ engine is the GPL v3 OLPC release of Micropolis (originally
Electronic Arts Inc., 1989–2007). This wrapper and search code are
distributed under the same license — see [LICENSE](LICENSE).

## For collaborators

- **[SUMMARY.md](SUMMARY.md)** — cross-cutting comparison of every approach
  family (layout / ELM / tape / hybrid / closed-loop nets / random-prefix).
  Start here if you want one-page orientation.
- **[RESULTS.md](RESULTS.md)** — what works, what doesn't, current leaderboard.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — chronological log of every training
  run, with config, wall time, replay-verified result, and a one-line
  insight. Append a new row when you launch a long run.
- **[LAYOUTS.md](LAYOUTS.md)** — deep-dive on the layout-evolution
  approach (currently the global champion at cityPop ≈ 1,820).
- All saved archives carry their full config as `.npz` metadata, so any
  result can be reproduced or replayed without runtime args.

## Layout

```
slimcity.py        thin Python wrapper around the SWIG-built engine
policy.py          ConvPolicy: 740-param NumPy conv → (tool, x, y)
evaluate.py        rollout one policy; returns (fitness, measures)
qd_train.py        CMA-ME training loop using pyribs GridArchive
plot_archive.py    text-art view of the saved archive
record.py          render a sim rollout as an animated GIF
random_smoke.py    quick CLI smoke test (random tools, prints trajectory)
visualize.py       ASCII map viewer (terminal animation)
viewer.py          tkinter live viewer (fragile on macOS Aqua Tk — use record.py)

engine/            C++ Micropolis source + SWIG bindings + assets
  src/             engine source
  swig/            Python bindings
  res/             runtime resources
  cities/          preset .cty files
  images/          XPM tile sprites (unused by our renderers)
  Makefile         builds the SWIG bindings
  setup.py         builds the C extension
```

## Setup

System Python 3.9 (the SWIG extension is built for `cpython-39-darwin.so`).

```bash
python3 -m pip install --user numpy ribs Pillow
```

### Build the engine

```bash
cd engine
make swig                                                    # generate SWIG wrappers
python3 setup.py build_ext --inplace --build-temp build/temp --build-lib build
cd ..
```

`engine/build/_micropolisengine.cpython-39-darwin.so` should now exist.

## Quick start

```bash
# Smoke-test the engine + wrapper
python3 random_smoke.py

# Run a tiny QD smoke (5 generations, ~10 seconds)
python3 qd_train.py --gens 5 --emitters 1 --batch 8

# Run a real QD experiment
python3 qd_train.py --gens 200 --emitters 5 --batch 20 --save archive.npz

# Same, parallelized across 8 worker processes (~4x faster on the M4)
python3 qd_train.py --gens 200 --emitters 5 --batch 20 --workers 8

# Look at the archive
python3 plot_archive.py archive.npz

# Watch a policy in action — saves an animated GIF macOS Preview can play
python3 record.py --mode policy --archive archive.npz --frames 200 --out elite.gif
open elite.gif
```

## Defaults

- **Empty map.** Every script starts from a blank generated map (engine
  `generateSomeCity` + `clearMap`). To load a preset city instead:
  `--city engine/cities/radial.cty`.
- **Fitness:** delta in `cityPop` (the in-game displayed population).
- **QD descriptors:** `road_frac` (roads as share of built tiles, [0, 1])
  and `ind_share` (industry as share of zoned tiles, [0, 1]).
- **Archive:** 20×20 GridArchive over `(road_frac, ind_share)`.

## Notes

- macOS system Tk 8.5 is buggy enough that `viewer.py` (live GUI) doesn't
  reliably render the canvas. Use `record.py` for visualizations. A
  pygame-based viewer would be the right path for true interactivity.
- The engine is fixed at 120×100 tiles (compile-time constant). All
  observation/policy code assumes this.
- Random policies on empty maps produce fitness=0 — the policy has to
  *discover* coordinated R/C/I + power + roads to grow anything. QD is a
  good match because the road/industry axes spread elites across
  fundamentally different city shapes.
