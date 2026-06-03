# fitness=407.1 cityPop=400 measures=(1.0, 0.0) origin=seed

def build_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    plan = []
    # Coal plant just left of the spine so it powers the wire column.
    plan.append((Tool.COALPOWER, cx - 2, cy))
    # Vertical infrastructure spine: wire at column cx, road at column cx+1.
    for dy in range(-12, 13):
        plan.append((Tool.WIRE, cx, cy + dy))
        plan.append((Tool.ROAD, cx + 1, cy + dy))
    # Residential zones flanking the spine every 3 rows. Centers at cx-3 (left)
    # and cx+3 (right) put each 3x3 zone's inner edge against wire/road.
    for dy in range(-12, 13, 3):
        plan.append((Tool.RESIDENTIAL, cx - 3, cy + dy))
        plan.append((Tool.RESIDENTIAL, cx + 3, cy + dy))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None                      # plan finished: place nothing
    state["i"] = i + 1
    return plan[i]                       # (tool, x, y); coords clamped if needed

