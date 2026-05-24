"""Live graphical viewer for MicropolisEnv (pure tkinter, macOS-Aqua-safe).

A real window with a colored tile grid that updates as the sim runs.
Play/Pause/Step/Reset controls, mode selection (idle/random/policy),
speed slider, last-action crosshair, and a stats panel.

Usage:
    python3 viewer.py
    python3 viewer.py --mode random --autoplay
    python3 viewer.py --archive archive.npz   # drive with best elite from a QD run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import tkinter as tk

import numpy as np

from slimcity import MicropolisEnv, Tool, WORLD_W, WORLD_H


# ---------- tile -> color lookup table ----------

def _build_color_table() -> np.ndarray:
    tbl = np.zeros((1024, 3), dtype=np.uint8)
    tbl[:] = (210, 200, 165)  # default: sandy tan empty land

    def shade(lo, hi, base, dark):
        n = hi - lo + 1
        for i in range(n):
            f = i / max(1, n - 1)
            tbl[lo + i] = (
                int(base[0] * (1 - f) + dark[0] * f),
                int(base[1] * (1 - f) + dark[1] * f),
                int(base[2] * (1 - f) + dark[2] * f),
            )

    for i in range(2, 20):   tbl[i] = (60, 160, 200)    # water
    for i in range(21, 37):  tbl[i] = (40, 110, 50)     # trees
    for i in range(44, 48):  tbl[i] = (140, 100, 70)    # rubble
    for i in range(48, 52):  tbl[i] = (80, 140, 180)    # flood
    for i in range(56, 64):  tbl[i] = (235, 90, 30)     # fire

    shade(64, 78,   (160, 160, 160), (90, 90, 90))      # roads
    shade(79, 95,   (120, 120, 120), (70, 70, 70))
    shade(96, 111,  (210, 200, 100), (170, 150, 60))    # wires
    shade(112, 175, (140, 140, 160), (90, 90, 110))
    shade(176, 206, (110, 90, 70),   (80, 60, 50))      # rails

    shade(240, 422, (180, 230, 180), (40, 140, 60))     # residential
    shade(423, 611, (180, 200, 240), (40, 80, 200))     # commercial
    shade(612, 692, (230, 220, 140), (170, 140, 30))    # industrial
    shade(693, 708, (90, 90, 140),   (60, 60, 110))     # seaport
    shade(709, 744, (190, 190, 200), (140, 140, 160))   # airport
    shade(745, 760, (90, 70, 70),    (50, 30, 30))      # coal plant
    shade(761, 770, (230, 90, 60),   (180, 50, 30))     # fire dept
    shade(771, 778, (100, 130, 220), (60, 90, 180))     # police dept
    shade(779, 810, (220, 130, 200), (170, 80, 150))    # stadium
    shade(811, 826, (200, 90, 200),  (140, 40, 140))    # nuclear plant
    return tbl


_COLOR_TBL = _build_color_table()
_TOOL_NAMES = {v: k for k, v in vars(Tool).items() if not k.startswith("_")}


class Viewer:
    def __init__(self, args):
        self.args = args
        here = os.path.dirname(os.path.abspath(__file__))
        self.city_path = os.path.join(here, args.city) if args.city else None

        self.env = MicropolisEnv(seed=args.seed, load_city=self.city_path)
        self.env.tick(args.warmup)

        self.rng = np.random.default_rng(args.seed)
        self.frame_count = 0
        self.running = False
        self.last_xy: tuple[int, int] | None = None
        self.last_tool: int | None = None

        self.policy = None
        self.theta = None
        self._load_policy()

        fd, self._ppm_path = tempfile.mkstemp(suffix=".ppm", prefix="slimcity_")
        os.close(fd)

        self._build_ui()
        self._draw_frame()

    # ---- UI ----

    def _build_ui(self):
        root = tk.Tk()
        root.title("slimcity viewer")
        self.root = root

        scale = self.args.scale
        self.img_w = WORLD_W * scale
        self.img_h = WORLD_H * scale

        # Left: canvas in its own frame, packed left.
        canvas_frame = tk.Frame(root, padx=8, pady=8)
        canvas_frame.pack(side="left", fill="y")

        self.canvas = tk.Canvas(
            canvas_frame, width=self.img_w, height=self.img_h,
            bd=2, relief="solid", highlightthickness=0,
        )
        self.canvas.pack()

        # Diagnostic rectangle: bright magenta border + center cross.
        # If you see this, the canvas itself is rendering — only the image
        # would have failed. Removed on the first _draw_frame().
        self._diag_ids = [
            self.canvas.create_rectangle(
                2, 2, self.img_w - 2, self.img_h - 2,
                outline="magenta", width=3,
            ),
            self.canvas.create_text(
                self.img_w // 2, self.img_h // 2,
                text="loading...", fill="magenta",
                font=("Menlo", 24, "bold"),
            ),
        ]

        self.photo = None
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.crosshair_id = None

        # Right: panel packed right, fills vertically.
        panel = tk.Frame(root, padx=10, pady=10)
        panel.pack(side="right", fill="y")

        # Stats — explicit fg/bg to avoid white-on-white, but the WIDGET's bg
        # is the system default so the panel still looks native.
        self.stats_label = tk.Label(
            panel, text="(initializing)", font=("Menlo", 12),
            justify="left", anchor="nw", width=26,
            relief="sunken", padx=6, pady=6,
        )
        self.stats_label.pack(anchor="w", pady=(0, 10))

        tk.Label(panel, text="mode:", font=("Menlo", 11, "bold")).pack(anchor="w")
        self.mode_var = tk.StringVar(value=self.args.mode)
        modes = ["idle", "random"]
        if self.theta is not None:
            modes.append("policy")
        tk.OptionMenu(panel, self.mode_var, *modes).pack(anchor="w", fill="x", pady=(0, 8))

        tk.Label(panel, text="ticks per frame:", font=("Menlo", 11, "bold")).pack(anchor="w")
        self.ticks_var = tk.IntVar(value=self.args.ticks_per_frame)
        tk.Scale(panel, from_=1, to=400, orient="horizontal",
                 variable=self.ticks_var, length=200).pack(anchor="w", pady=(0, 4))

        tk.Label(panel, text="frame delay (ms):", font=("Menlo", 11, "bold")).pack(anchor="w")
        self.delay_var = tk.IntVar(value=int(self.args.delay * 1000))
        tk.Scale(panel, from_=0, to=500, orient="horizontal",
                 variable=self.delay_var, length=200).pack(anchor="w", pady=(0, 10))

        btn_row1 = tk.Frame(panel)
        btn_row1.pack(anchor="w", fill="x")
        self.play_btn = tk.Button(btn_row1, text="Play", width=8, command=self._toggle)
        self.play_btn.pack(side="left", padx=2)
        tk.Button(btn_row1, text="Step", width=8, command=self._step_once).pack(side="left", padx=2)

        btn_row2 = tk.Frame(panel)
        btn_row2.pack(anchor="w", fill="x", pady=(4, 10))
        tk.Button(btn_row2, text="Reset", width=8, command=self._reset).pack(side="left", padx=2)
        tk.Button(btn_row2, text="Quit", width=8, command=self._quit).pack(side="left", padx=2)

        tk.Label(panel, text="legend:", font=("Menlo", 11, "bold")).pack(anchor="w")
        legend_items = [
            ("#2b8c3c", "residential"),
            ("#2a58c8", "commercial"),
            ("#bb9220", "industrial"),
            ("#777777", "road / rail"),
            ("#daca64", "wire"),
            ("#3ca036", "trees"),
            ("#3ca0c8", "water"),
            ("#8c2a8c", "power plant"),
            ("#d2c8a5", "empty land"),
        ]
        for color, name in legend_items:
            row_frame = tk.Frame(panel)
            row_frame.pack(anchor="w", fill="x", pady=1)
            sw = tk.Label(row_frame, text="    ", bg=color, width=3, relief="solid", bd=1)
            sw.pack(side="left", padx=(0, 6))
            tk.Label(row_frame, text=name, font=("Menlo", 10)).pack(side="left")

        root.bind("<space>", lambda e: self._toggle())
        root.bind("s", lambda e: self._step_once())
        root.bind("r", lambda e: self._reset())
        root.bind("q", lambda e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        # Diagnostic: report what we actually built.
        root.update_idletasks()
        print(f"[viewer] canvas requested {self.img_w}x{self.img_h}, "
              f"actual {self.canvas.winfo_reqwidth()}x{self.canvas.winfo_reqheight()}",
              file=sys.stderr)
        print(f"[viewer] root size {root.winfo_reqwidth()}x{root.winfo_reqheight()}",
              file=sys.stderr)

    def _load_policy(self):
        if not self.args.archive:
            return
        if not os.path.exists(self.args.archive):
            print(f"warning: archive {self.args.archive} not found, ignoring",
                  file=sys.stderr)
            return
        d = np.load(self.args.archive)
        if len(d["objectives"]) == 0:
            print("warning: archive is empty", file=sys.stderr)
            return
        idx = int(np.argmax(d["objectives"]))
        self.theta = d["solutions"][idx].astype(np.float32)
        from policy import ConvPolicy
        self.policy = ConvPolicy()
        self.policy.set_params(self.theta)
        print(f"loaded elite from {self.args.archive}: "
              f"obj={float(d['objectives'][idx]):.0f}, "
              f"measures={d['measures'][idx]}")

    # ---- main loop ----

    def run(self):
        if self.args.autoplay:
            self.running = True
            self.play_btn.config(text="Pause")
        self._schedule_next()
        self.root.mainloop()

    def _schedule_next(self):
        if self.running:
            self._step_once()
        delay = max(1, self.delay_var.get()) if self.running else 80
        self.root.after(delay, self._schedule_next)

    def _step_once(self):
        mode = self.mode_var.get()
        if mode == "random":
            tool = self._sample_tool()
            x = int(self.rng.integers(0, WORLD_W))
            y = int(self.rng.integers(0, WORLD_H))
            self.env.place(tool, x, y)
            self.last_xy = (x, y)
            self.last_tool = tool
        elif mode == "policy" and self.policy is not None:
            tool, x, y = self.policy.act(self.env.get_map())
            self.env.place(tool, x, y)
            self.last_xy = (x, y)
            self.last_tool = tool
        else:
            self.last_xy = None
            self.last_tool = None
        self.env.tick(self.ticks_var.get())
        self.frame_count += 1
        self._draw_frame()

    def _toggle(self):
        self.running = not self.running
        self.play_btn.config(text="Pause" if self.running else "Play")

    def _reset(self):
        self.running = False
        self.play_btn.config(text="Play")
        self.env = MicropolisEnv(seed=self.args.seed, load_city=self.city_path)
        self.env.tick(self.args.warmup)
        self.frame_count = 0
        self.last_xy = None
        self.last_tool = None
        self.rng = np.random.default_rng(self.args.seed)
        if self.theta is not None and self.policy is not None:
            self.policy.set_params(self.theta)
        self._draw_frame()

    def _quit(self):
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        try:
            os.remove(self._ppm_path)
        except OSError:
            pass

    # ---- rendering ----

    def _draw_frame(self):
        # Clear the diagnostic overlay the first time we draw something real.
        if self._diag_ids:
            for i in self._diag_ids:
                self.canvas.delete(i)
            self._diag_ids = []

        tile_map = self.env.get_map()
        rgb = _COLOR_TBL[tile_map]
        scale = self.args.scale
        if scale != 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        H, W = rgb.shape[:2]

        with open(self._ppm_path, "wb") as f:
            f.write(f"P6\n{W} {H}\n255\n".encode("ascii"))
            f.write(rgb.tobytes())

        try:
            new_photo = tk.PhotoImage(file=self._ppm_path)
        except tk.TclError as e:
            print(f"[viewer] PhotoImage load failed: {e}", file=sys.stderr)
            return
        self.canvas.itemconfig(self.canvas_image_id, image=new_photo)
        self.photo = new_photo  # keep reference

        # Crosshair on last action
        if self.crosshair_id is not None:
            self.canvas.delete(self.crosshair_id)
            self.crosshair_id = None
        if self.last_xy is not None:
            x, y = self.last_xy
            cx, cy = x * scale + scale // 2, y * scale + scale // 2
            r = max(scale * 2, 6)
            self.crosshair_id = self.canvas.create_rectangle(
                cx - r, cy - r, cx + r, cy + r, outline="#ff3030", width=2,
            )

        self._update_stats()

    def _update_stats(self):
        s = self.env.stats
        if self.last_tool is not None and self.last_xy is not None:
            action = f"{_TOOL_NAMES.get(self.last_tool, '?')} @{self.last_xy}"
        else:
            action = "(none)"
        text = (
            f"frame      {self.frame_count}\n"
            f"city time  {s.city_time}\n"
            f"cityPop    {s.city_pop}\n"
            f"R/C/I      {s.res_pop}/{s.com_pop}/{s.ind_pop}\n"
            f"funds      {s.funds}\n"
            f"pollution  {s.pollution_avg}\n"
            f"traffic    {s.traffic_avg}\n"
            f"crime      {s.crime_avg}\n"
            f"last act   {action}"
        )
        self.stats_label.config(text=text)

    def _sample_tool(self) -> int:
        pool = [
            (Tool.RESIDENTIAL,   8),
            (Tool.COMMERCIAL,    4),
            (Tool.INDUSTRIAL,    4),
            (Tool.ROAD,         10),
            (Tool.WIRE,          4),
            (Tool.PARK,          2),
            (Tool.RAILROAD,      1),
            (Tool.POLICESTATION, 1),
            (Tool.FIRESTATION,   1),
        ]
        tools, weights = zip(*pool)
        p = np.array(weights, dtype=np.float64)
        p = p / p.sum()
        return int(self.rng.choice(tools, p=p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="",
                    help="preset .cty to load; default '' = empty map")
    ap.add_argument("--mode", choices=["idle", "random", "policy"], default="idle")
    ap.add_argument("--archive", default=None,
                    help="npz archive (from qd_train.py) to load policy params from")
    ap.add_argument("--scale", type=int, default=6, help="pixels per tile")
    ap.add_argument("--ticks-per-frame", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.05)
    ap.add_argument("--autoplay", action="store_true")
    args = ap.parse_args()

    Viewer(args).run()


if __name__ == "__main__":
    main()
