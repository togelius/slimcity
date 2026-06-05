"""Investigate the evolved ELM policies: closed-loop?, strategies, diversity.

Three analyses over the archive (.npz) + generation log (.jsonl):

1. CLOSED-LOOP — static (does the source reference obs.tile_map / masks /
   stats?) AND dynamic (counterfactual: at each step, does the policy return a
   different action when fed a BLINDED obs vs the real obs? If never, the
   policy ignores what it observes => open-loop).

2. STRATEGY — motif detection (precomputed plan? #power plants ~ #neighborhoods?
   wires/roads? reactive placement?).

3. DIVERSITY — behavior-space cells vs distinct code: pairwise source similarity
   clustering, plus champion lineage trace through the generation log.

Run: /usr/bin/python3 -m elm.analyze [results/elm_resind_sonnet_v2]
"""

from __future__ import annotations

import copy
import difflib
import json
import re
import sys
from collections import Counter

import numpy as np

from elm.archive import MapElitesArchive
from elm.sandbox import Obs, compile_policy, parse_action
from slimcity import MicropolisEnv, WORLD_W, WORLD_H

MAP_REF = re.compile(r"obs\.tile_map|\b(res_mask|com_mask|ind_mask|road_mask|"
                     r"wire_mask|plant_mask|empty_mask)\s*\(")
STATS_REF = re.compile(r"obs\.(city_pop|res_pop|com_pop|ind_pop|funds|powered_zones)")
STEP_REF = re.compile(r"obs\.(step|n_steps)")


def static_class(src: str) -> str:
    if MAP_REF.search(src):
        return "reactive-map"
    if STATS_REF.search(src):
        return "reactive-stats"
    if STEP_REF.search(src):
        return "time-indexed"
    return "open-loop"


def dynamic_reacts(src: str, n_actions=120, ticks=100, seed=42) -> tuple[bool, int]:
    """Counterfactual probe. Returns (reacts?, n_steps_where_blinded_action_differs).
    At each real step we also call act() on a deepcopy of state with a BLINDED
    obs (zeroed tile_map + zeroed stats). If the action ever differs, the policy
    is genuinely using its observation."""
    try:
        act = compile_policy(src)
    except Exception:
        return False, -1
    env = MicropolisEnv(seed=seed)
    baseline = env.stats
    state: dict = {}
    diffs = 0
    for step in range(n_actions):
        tm = env.get_map()
        s = env.stats
        obs_real = Obs(tm, step, n_actions, s.city_pop, s.res_pop, s.com_pop,
                       s.ind_pop, s.funds, env.engine.poweredZoneCount)
        blind = Obs(np.zeros_like(tm), step, n_actions, 0, 0, 0, 0, 0, 0)
        st_real = copy.deepcopy(state)
        try:
            a_real = act(obs_real, st_real)
            st_blind = copy.deepcopy(state)
            a_blind = act(blind, st_blind)
        except Exception:
            break
        if parse_action(a_real) != parse_action(a_blind):
            diffs += 1
        state = st_real
        p = parse_action(a_real)
        if p is not None:
            env.place(*p)
        env.tick(ticks)
    return diffs > 0, diffs


def norm_src(src: str) -> str:
    out = []
    for ln in src.splitlines():
        ln = re.sub(r"#.*$", "", ln).rstrip()
        if ln.strip():
            out.append(ln)
    return "\n".join(out)


def cluster(sources, thresh=0.85):
    reps, labels = [], []
    for s in sources:
        ns = norm_src(s)
        best, bi = 0.0, -1
        for i, r in enumerate(reps):
            ratio = difflib.SequenceMatcher(None, ns, r).ratio()
            if ratio > best:
                best, bi = ratio, i
        if best >= thresh:
            labels.append(bi)
        else:
            reps.append(ns)
            labels.append(len(reps) - 1)
    return labels


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results/elm_resind_sonnet_v2"
    a = MapElitesArchive.load(base + ".npz")
    elites = sorted(a.cells.values(), key=lambda e: -e.city_pop)
    n = len(elites)
    print(f"archive: {n} elites, best cityPop {elites[0].city_pop}\n")

    # ---- 1. closed-loop ----
    print("=" * 64)
    print("1. CLOSED-LOOP?  (static class + dynamic counterfactual probe)")
    print("=" * 64)
    stat = Counter()
    dyn_react = 0
    dyn_tested = 0
    rows = []
    for e in elites:
        sc = static_class(e.source)
        stat[sc] += 1
        reacts, diffs = dynamic_reacts(e.source)
        if diffs >= 0:
            dyn_tested += 1
            if reacts:
                dyn_react += 1
        rows.append((e.city_pop, sc, reacts, diffs, e.origin, e.iteration))
    print("static classification:", dict(stat))
    print(f"dynamic: {dyn_react}/{dyn_tested} policies change their action when "
          f"the observation is blinded (i.e. genuinely closed-loop).")
    print("\nper-policy (top 15 by cityPop):")
    print(f"{'pop':>5} {'static':>14} {'reacts':>7} {'diff_steps':>10} {'origin':>10} it")
    for pop, sc, reacts, diffs, org, it in rows[:15]:
        print(f"{pop:5d} {sc:>14} {str(reacts):>7} {diffs:>10} {org:>10} {it}")

    # ---- 2. strategy motifs ----
    print("\n" + "=" * 64)
    print("2. STRATEGIES (motif detection)")
    print("=" * 64)
    motifs = Counter()
    for e in elites:
        s = e.source
        if "build_plan" in s or ("plan" in s and "state[" in s):
            motifs["precomputed-plan"] += 1
        nc = s.count("COALPOWER")
        motifs["multi-plant(>=2)" if nc >= 2 else f"plants={nc}"] += 1
        if "NUCLEAR" in s:
            motifs["uses-nuclear"] += 1
        if "Tool.ROAD" in s:
            motifs["uses-road"] += 1
        if "Tool.WIRE" in s:
            motifs["uses-wire"] += 1
        if "PARK" in s:
            motifs["uses-park"] += 1
    for k, v in sorted(motifs.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d}/{n}  {k}")
    coal = [e.source.count("COALPOWER") for e in elites]
    print(f"  power-plants per policy: min={min(coal)} max={max(coal)} "
          f"mean={np.mean(coal):.1f}")

    # ---- 3. diversity ----
    print("\n" + "=" * 64)
    print("3. DIVERSITY (behavior cells vs distinct code)")
    print("=" * 64)
    labels = cluster([e.source for e in elites])
    nclust = len(set(labels))
    sizes = Counter(labels)
    print(f"behavior cells filled: {n}")
    print(f"distinct code clusters (>=0.85 similarity): {nclust}")
    print(f"  cluster sizes: {sorted(sizes.values(), reverse=True)}")
    uniq = len(set(norm_src(e.source) for e in elites))
    print(f"  exact-unique (normalized) sources: {uniq}/{n}")

    # Reconstruct full genealogy from the generation log by replaying eid
    # assignment (archive.add assigns the next eid to every IMPROVED genome,
    # in chronological order) — this recovers even ancestors later overwritten.
    try:
        recs = [json.loads(l) for l in open(base + "_generations.jsonl")]
    except FileNotFoundError:
        recs = []
    rec_by_eid, eid = {}, 0
    for r in recs:
        if r.get("event") == "eval" and r.get("improved"):
            r["_eid"] = eid
            rec_by_eid[eid] = r
            eid += 1

    def trace(start_eid):
        chain, cur, seen = [], rec_by_eid.get(start_eid), set()
        while cur is not None and cur.get("_eid") not in seen:
            seen.add(cur["_eid"])
            chain.append(cur)
            par = (cur.get("parents") or [None])[0]
            cur = rec_by_eid.get(par)
        return chain

    champ = elites[0]
    chain = trace(champ.eid)
    print(f"\nchampion lineage (cityPop {champ.city_pop}, it {champ.iteration}):")
    for c in chain:
        print(f"  it {c['iteration']:>4}  {c['origin']:>9}  pop={c['city_pop']:>5}  "
              f"m=({c['measures'][0]:.2f},{c['measures'][1]:.2f})")
    root = chain[-1] if chain else None
    print(f"  chain length {len(chain)}; root origin = "
          f"{root['origin'] if root else '?'}")

    # lineage depth distribution + how many survivors trace back to the seed
    depths, to_seed = [], 0
    for e in elites:
        ch = trace(e.eid)
        depths.append(len(ch))
        if ch and ch[-1]["origin"] == "seed":
            to_seed += 1
    if depths:
        print(f"\nlineage depth over {len(elites)} survivors: "
              f"min={min(depths)} max={max(depths)} mean={np.mean(depths):.1f}")
        print(f"survivors whose traceable lineage reaches the seed: {to_seed}/{len(elites)}")
    print(f"elite origins: {dict(Counter(e.origin for e in elites))}")


if __name__ == "__main__":
    main()
