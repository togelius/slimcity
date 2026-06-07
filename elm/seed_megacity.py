"""Hand-designed champion city-building policy ("megacity").

Open-loop blueprint distilled from a strategy-guide-informed empirical search on
this engine (see ELM_DIVERSE_RUN.md / the citylab sweeps). Key findings it
exploits:
  * power is THE bottleneck — zones conduct power to neighbours, but a plant has
    finite supply, so MANY distributed plants are needed to power a dense block;
  * pack 3x3 zones edge-to-edge (3-tile pitch) for maximum density, with a WIRE
    spine every 4th zone-column to inject/distribute power (wires conduct, roads
    don't); plants sit on the spines;
  * pure RESIDENTIAL maximises population here; INDUSTRIAL pollutes and craters
    growth; a little commercial barely helps;
  * NUCLEAR plants (low pollution, high supply) beat coal on mean population
    (coal is steadier but lower).
Reaches a sustained cityPop ~14,500 (peak ~26,700) vs the evolved record 4,460.
Needs a large action budget (~1,632 placements) + long stabilisation.
"""

SEED_MEGACITY = '''
def build_plan():
    # Dense residential block: 36x30 zones on a 3-tile pitch, wire spine every
    # 4th column, 12 nuclear plants distributed on the spines to power it all.
    x0, y0, nx, ny, we, npl = 3, 3, 36, 30, 4, 12
    plan = []
    spine = []
    for i in range(nx):
        cx = x0 + 1 + 3 * i
        if i % we == 0:                       # wire spine column (conducts power)
            for j in range(3 * ny):
                plan.append((Tool.WIRE, cx - 1, y0 + j))
            spine.append(cx - 1)
        else:                                 # column of residential zones
            for j in range(ny):
                plan.append((Tool.RESIDENTIAL, cx, y0 + 1 + 3 * j))
    for p in range(npl):                       # plants first (power before zones)
        col = spine[p % len(spine)]
        ry = y0 + 1 + 3 * int((p + 0.5) * ny / npl)
        plan.insert(0, (Tool.NUCLEARPOWER, col, ry))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    if i >= len(state["plan"]):
        return None                            # built: idle and let it grow
    state["i"] = i + 1
    return state["plan"][i]
'''
