"""Gather raw material for a DEEP qualitative analysis of evolved policies.

Selects the champion + a diverse sample (most reactive, most divergent, best
open-loop, most industrial / residential, best reactive) and for each dumps:
behavior measures, code length, static closed-loop class + motif flags, the
city it builds (ascii render + R/C/I stats), and the full source.

A human/LLM reads this dump and writes the prose interpretation (what does it
do, how does it decide, what city does it build). Markdown to stdout.

Run: /usr/bin/python3 -m elm.deep_analysis results/elm_clind_react [n_actions ticks]
"""

from __future__ import annotations

import sys

import numpy as np

from elm.archive import MapElitesArchive
from elm.analyze import static_class, MAP_REF, STATS_REF
from elm.evaluate_code import eval_code


def _motifs(src: str) -> str:
    flags = []
    if "build_plan" in src or ("plan" in src and "state[" in src and "append" in src):
        flags.append("precomputed-plan")
    if MAP_REF.search(src):
        flags.append("reads-map")
    if STATS_REF.search(src):
        flags.append("reads-stats")
    nc = src.count("COALPOWER"); nn = src.count("NUCLEAR")
    flags.append(f"plants={nc}c{('+%dn' % nn) if nn else ''}")
    if "Tool.WIRE" in src: flags.append("wire")
    if "Tool.ROAD" in src: flags.append("road")
    if "while " in src: flags.append("has-loop-in-act")
    return ", ".join(flags)


def _pick(archive):
    E = list(archive.cells.values())
    react = [e for e in E if e.measures[0] > 0 or e.measures[1] > 0]
    openl = [e for e in E if e.measures[0] == 0 and e.measures[1] == 0]
    cand = {}
    def add(label, e):
        if e is not None and e.eid not in cand:
            cand[e.eid] = (label, e)
    add("champion (max cityPop)", max(E, key=lambda e: e.city_pop, default=None))
    if react:
        add("best reactive", max(react, key=lambda e: e.city_pop))
        add("most counterfactual-sensitive (max cf)", max(react, key=lambda e: e.measures[0]))
        add("most trajectory-divergent (max div)", max(react, key=lambda e: e.measures[1]))
    if openl:
        add("best open-loop", max(openl, key=lambda e: e.city_pop))
    add("most industrial", max(E, key=lambda e: e.measures[-1]))
    return list(cand.values())


def build(base: str, n_actions=120, ticks=100) -> str:
    a = MapElitesArchive.load(base + ".npz")
    L = [f"# Deep analysis — {base.split('/')[-1]}",
         f"archive: {len(a.cells)} cells, qd={a.qd_score:.0f}, "
         f"best cityPop={a.best.city_pop}", ""]
    for label, e in _pick(a):
        src = e.source
        nlines = len(src.strip().splitlines())
        nchars = len(src)
        r = eval_code(src, seed=42, n_actions=n_actions, ticks_per_action=ticks,
                      measures_mode="res_ind", collect_render=True)
        cf, div = e.measures[0], e.measures[1]
        ind = e.measures[2] if len(e.measures) > 2 else e.measures[1]
        L.append(f"## {label}")
        L.append(f"- cityPop={r.city_pop} (R={r.res_pop} C={r.com_pop} I={r.ind_pop}) "
                 f"| stored cityPop={e.city_pop} | origin={e.origin} iter={e.iteration}")
        L.append(f"- reactivity: cf_sensitivity={cf:.2f}, traj_divergence={div:.2f}, "
                 f"ind_share={ind:.2f}  [{static_class(src)}]")
        L.append(f"- code: {nlines} lines, {nchars} chars | motifs: {_motifs(src)}")
        if r.render:
            L.append("```\n" + r.render + "\n```")
        L.append("```python\n" + src.strip() + "\n```")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "results/elm_clind_react"
    na = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    tk = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    md = build(base, na, tk)
    out = base + "_DEEP.md"
    open(out, "w").write(md + "\n")
    print(md)
