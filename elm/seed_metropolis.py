"""Hand-designed champion v2 ("metropolis") — the road+land-value strategy.

Distilled from a second empirical search after finding the megacity's weakness:
zones were power-served but ROAD-STARVED, capping them at low density. Adding
road access on every zone row ~doubled population; sparse PARKS (land value) on
top of roads added more (parks only help once roads let zones densify). Pure
residential still wins; INDUSTRIAL pollution craters growth.

Recipe: 30x28 dense residential block on a 3-tile pitch, WIRE spine every 4th
column (power) with 18 distributed NUCLEAR plants, a ROAD row on every zone row
(access), every 5th cell a PARK (land value), plus police (crime). Sustained
cityPop ~36,000 (5-seed) vs the megacity's ~14,400 and the evolved record 4,460.
~3,836 placements; long stabilisation.
"""

SEED_METROPOLIS = '''
def build_plan():
    x0, y0, nx, ny, we, npl, park_every = 2, 2, 30, 28, 4, 18, 5
    plan = []; spine = []; zc = []
    for i in range(nx):
        cx = x0 + 1 + 3 * i
        if i % we == 0:                              # wire spine column (power)
            for j in range(3 * ny):
                plan.append((Tool.WIRE, cx - 1, y0 + j))
            spine.append(cx - 1)
        else:
            for j in range(ny):
                zc.append((cx, y0 + 1 + 3 * j))       # residential/park cell
    for idx, (cx, cy) in enumerate(zc):
        plan.append((Tool.PARK if idx % park_every == 0 else Tool.RESIDENTIAL, cx, cy))
    for j in range(ny):                              # ROAD row on every zone row (access)
        ry = y0 + 3 * j
        for xx in range(x0, x0 + 3 * nx):
            plan.append((Tool.ROAD, xx, ry))
    for p in range(npl):                             # plants first (power before zones)
        col = spine[p % len(spine)]
        ry = y0 + 1 + 3 * int((p + 0.5) * ny / npl)
        plan.insert(0, (Tool.NUCLEARPOWER, col, ry))
    for s in range(10):                              # police: lower crime -> land value
        plan.append((Tool.POLICESTATION, x0 + (s * 13) % (3 * nx), y0 + (s * 7) % (3 * ny)))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan(); state["i"] = 0
    i = state["i"]
    if i >= len(state["plan"]):
        return None
    state["i"] = i + 1
    return state["plan"][i]
'''
