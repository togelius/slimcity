# MEGACITY — best hand-designed city policy
# sustained cityPop ~14400 (trough 5100, peak 26720, 5 seeds); placements=1632
# eval: n_actions>=1632, long stabilisation (~100-150k ticks)
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
