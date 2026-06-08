"""Summarize an ELM run: archive (.npz) + full generation log (.jsonl).

Writes a markdown report to <base>_SUMMARY.md and prints it. Used for the final
readout and as the email body.

Usage: /usr/bin/python3 -m elm.report [results/elm_resind_sonnet_v2]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from elm.archive import MapElitesArchive


def build_report(base: str) -> str:
    npz = base + ".npz"
    jsonl = base + "_generations.jsonl"
    a = MapElitesArchive.load(npz)

    recs = []
    try:
        recs = [json.loads(l) for l in open(jsonl)]
    except FileNotFoundError:
        pass
    evals = [r for r in recs if r.get("event") == "eval"]
    events = Counter(r.get("event") for r in recs)
    attempts = events.get("eval", 0) + events.get("op_error", 0) + events.get("invalid", 0)
    grew = [r for r in evals if r.get("city_pop", 0) > 0]
    last_iter = max([r.get("iteration", -1) for r in evals] + [-1])

    b = a.best
    top = sorted(a.cells.values(), key=lambda e: -e.city_pop)[:10]
    hist = a.history

    # Behavior-axis labels: 3-D archive is the closed-loop descriptor (cl_ind),
    # 2-D is the (res_share, ind_share) family.
    ndim = len(a.dims)
    axes = (["cf_sensitivity", "traj_divergence", "ind_share"] if ndim == 3
            else ["res_share", "ind_share"])

    def mstr(m):
        return ", ".join(f"{axes[i]}={m[i]:.2f}" for i in range(len(m)))

    L = []
    L.append(f"# SlimCity ELM run — {base.split('/')[-1]}")
    L.append("")
    L.append("Quality-Diversity (MAP-Elites) over **Python-code genomes**, with "
             "**Claude (Sonnet) as the mutation/crossover operator**. Each genome "
             "is a closed-loop `act(obs, state)` tile-placement policy; fitness = "
             "cityPop growth (dense-shaped), behavior = (residential share, "
             "industrial share); fitness is the mean of 5 independent rolls "
             "(the engine eval is ~19% noisy per process).")
    L.append("")
    L.append("## Headline")
    if b:
        L.append(f"- **Best city: cityPop {b.city_pop}** (robust mean), fitness "
                 f"{b.fitness:.0f}, found at iter {b.iteration} via **{b.origin}**, "
                 f"behavior ({mstr(b.measures)}).")
        L.append(f"- For comparison, the prior CMA-ME state of the art was "
                 f"cityPop 1,680 (layout genome) / 1,120 (action tape). "
                 f"{'**ELM beats it.**' if b.city_pop > 1680 else ''}")
    L.append(f"- Archive: **{len(a.cells)} cells filled** "
             f"({100*a.coverage:.1f}% of {int(np.prod(a.dims))}), "
             f"QD-score **{a.qd_score:.0f}**.")
    L.append(f"- Iterations completed: **{last_iter}**. "
             f"Genomes that grew a city (pop>0): **{len(grew)}/{len(evals)}** evaluated.")
    L.append("")
    L.append("## Operator efficiency")
    L.append(f"- eval={events.get('eval',0)}  op_error={events.get('op_error',0)}  "
             f"invalid={events.get('invalid',0)}  (of {attempts} attempts)")
    if attempts:
        L.append(f"- operator failure rate: "
                 f"{100*events.get('op_error',0)/attempts:.0f}% op-error, "
                 f"{100*events.get('invalid',0)/attempts:.0f}% invalid.")
    L.append("")
    L.append("## Top cities (robust mean over 5 rolls)")
    L.append("")
    L.append("| rank | cityPop | fitness | " + " | ".join(axes) + " | origin | iter |")
    L.append("|---|---|---|" + "---|" * len(axes) + "---|---|")
    for i, e in enumerate(top, 1):
        mcols = " | ".join(f"{e.measures[k]:.2f}" for k in range(len(e.measures)))
        L.append(f"| {i} | {e.city_pop} | {e.fitness:.0f} | {mcols} | "
                 f"{e.origin} | {e.iteration} |")
    L.append("")
    L.append("## QD-score / coverage progress")
    if hist:
        L.append("")
        L.append("| iter | filled | coverage | qd_score | best_pop |")
        L.append("|---|---|---|---|---|")
        step = max(1, len(hist) // 12)
        for h in hist[::step] + ([hist[-1]] if hist[-1] not in hist[::step] else []):
            L.append(f"| {h['iteration']} | {h['filled']} | {h['coverage']:.3f} | "
                     f"{h['qd_score']:.0f} | {h['best_city_pop']} |")
    L.append("")
    L.append("## Origin of elites")
    org = Counter(e.origin for e in a.cells.values())
    L.append("- " + ", ".join(f"{k}={v}" for k, v in org.items()))
    L.append("")
    L.append("## Champion policy (source)")
    if b:
        L.append("```python")
        L.append(b.source.strip())
        L.append("```")
    L.append("")
    L.append(f"Artifacts: `{npz}` (archive, every cell-winner's full source), "
             f"`{jsonl}` (every genome generated), `{base}_champion.py`.")
    return "\n".join(L)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "results/elm_resind_sonnet_v2"
    md = build_report(base)
    out = base + "_SUMMARY.md"
    with open(out, "w") as f:
        f.write(md + "\n")
    # print a short head for the console
    print(md[:1500])
    print(f"\n... [full report written to {out}]")
