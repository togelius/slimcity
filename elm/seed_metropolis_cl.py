"""Closed-loop rewrite of the open-loop `metropolis` blueprint.

Same target layout (dense R + nuclear + wire spines + road-per-row + parks) but
act() READS obs each step and places the first target tile not yet present on
the observed map (cf_sensitivity~1.0 -> genuinely closed-loop, survives the
--min-reactivity filter). It reaches sustained cityPop ~3,300 at a big budget
(~4,300 actions) -- ~10x BELOW the open-loop metropolis (~37,000). The gap is
the engine's auto-bulldoze: the open-loop result depends on a precise one-pass
placement ORDER (place everything, then stabilise), which reactive re-scanning
disrupts (roads clip zones/wires, tiles get re-placed, zones never settle).
Still, ~3,300 beats every EVOLVED closed-loop policy -- it is the closed-loop
ceiling on this engine.
"""

SEED_METROPOLIS_CL = '''
def build_plan():
    x0, y0, nx, ny, we, npl, park_every = 2, 2, 30, 28, 4, 18, 5
    plan = []; spine = []; zc = []; spcols = set()
    for i in range(nx):
        cx = x0 + 1 + 3 * i
        if i % we == 0:
            for j in range(3 * ny): plan.append((Tool.WIRE, cx - 1, y0 + j))
            spine.append(cx - 1); spcols.add(cx - 1)
        else:
            for j in range(ny): zc.append((cx, y0 + 1 + 3 * j))
    for idx, (cx, cy) in enumerate(zc):
        plan.append((Tool.PARK if idx % park_every == 0 else Tool.RESIDENTIAL, cx, cy))
    for j in range(ny):
        ry = y0 + 3 * j
        for xx in range(x0, x0 + 3 * nx):
            if xx not in spcols:                 # keep roads OFF the wire spines
                plan.append((Tool.ROAD, xx, ry))
    for p in range(npl):
        col = spine[p % len(spine)]; ry = y0 + 1 + 3 * int((p + 0.5) * ny / npl)
        plan.insert(0, (Tool.NUCLEARPOWER, col, ry))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan(); state["done"] = set()
    tm = obs.tile_map; done = state["done"]
    for k, (tool, x, y) in enumerate(state["plan"]):
        if k in done or not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            continue
        t = int(tm[y, x])
        present = (ROAD_RANGE[0] <= t <= ROAD_RANGE[1]) if tool == Tool.ROAD else (t != EMPTY_TILE)
        if not present:
            return (tool, x, y)
        done.add(k)                              # observed already-built -> mark done
    return None
'''
