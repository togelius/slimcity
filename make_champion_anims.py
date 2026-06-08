"""Render the best individual of each run family with the real tileset:
a full still PNG + an auto-cropped animated GIF (engine-driven tile animation).

Outputs to docs/champions/. Run:  /usr/bin/python3 make_champion_anims.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import render_micropolis as R

OUT = os.path.join(HERE, "docs", "champions")
os.makedirs(OUT, exist_ok=True)

# (label, kind, path)  — best individual per run family
CHAMPIONS = [
    ("elm_clind_long_15k",  "elm",     "results/elm_clind_long_best.py"),
    ("elm_clind_react_5k",  "elm",     "results/elm_clind_react_best.py"),
    ("elm_diverse_4k",      "elm",     "results/elm_diverse_overnight_best.py"),
    ("layout_weekend_2.7k", "archive", "results/weekend_layout_resind.npz"),
    ("tape600_1.1k",        "archive", "results/overnight_tape_600_varied.npz"),
]


def bbox(e):
    m = R._read_map(e)
    ys, xs = np.nonzero(m != 0)
    if not len(xs):
        return (0, 0, R.WORLD_W, R.WORLD_H)
    mgn = 3
    return (max(0, xs.min() - mgn), max(0, ys.min() - mgn),
            min(R.WORLD_W, xs.max() + 1 + mgn), min(R.WORLD_H, ys.max() + 1 + mgn))


def main():
    ts = R.Tileset()
    for label, kind, path in CHAMPIONS:
        if not os.path.exists(path):
            print(f"skip {label}: {path} missing"); continue
        print(f"== {label} ({kind}) ==")
        if kind == "elm":
            env, e = R._engine_for_elm(path)
        else:
            env, e = R._engine_for_archive(path)
        s = env.stats
        print(f"   cityPop={s.city_pop} R={s.res_pop} C={s.com_pop} I={s.ind_pop}")
        # full still
        R.render_still(e, ts, scale=1).save(os.path.join(OUT, f"{label}.png"))
        # auto-cropped animated GIF, scale to keep max dim ~1100px
        cb = bbox(e)
        w, h = (cb[2] - cb[0]) * 16, (cb[3] - cb[1]) * 16
        scale = max(1, min(4, 1100 // max(w, h)))
        R.render_gif(e, ts, os.path.join(OUT, f"{label}.gif"),
                     frames=16, sim_ticks_per_frame=2, anim_steps_per_frame=1,
                     crop=cb, scale=scale, fps=8)
        print(f"   wrote {label}.png + {label}.gif (crop {cb}, scale {scale})")


if __name__ == "__main__":
    main()
