# it89 robust_fit=1823 cityPop=1792 m=(0.6666666666666666, 0.11111111111111112)
def build_plan():
    plan = []
    
    # Strategy: Multiple compact neighborhoods, each with:
    # - Coal power plant
    # - Horizontal wire spine directly adjacent to zones
    # - Road lane for zone access
    # - Dense mixed R/C/I zoning above and below
    # Connect neighborhoods with wires for redundancy
    
    # Neighborhood 1: top-left area
    cx1, cy1 = 25, 20
    plan.append((Tool.COALPOWER, cx1 - 12, cy1))
    # Wire spine
    for dx in range(-10, 18):
        plan.append((Tool.WIRE, cx1 + dx, cy1))
    # Road just below wire
    for dx in range(-10, 18):
        plan.append((Tool.ROAD, cx1 + dx, cy1 + 1))
    # Residential row above wire (centers at cy1-3)
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 - 3))
    # Mixed commercial/industrial row below road (centers at cy1+4)
    for i, dx in enumerate(range(-9, 18, 3)):
        if i % 3 == 0:
            plan.append((Tool.COMMERCIAL, cx1 + dx, cy1 + 4))
        elif i % 3 == 1:
            plan.append((Tool.INDUSTRIAL, cx1 + dx, cy1 + 4))
        else:
            plan.append((Tool.COMMERCIAL, cx1 + dx, cy1 + 4))
    # Second residential row below
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 + 7))
    # Road for second row access
    for dx in range(-10, 18):
        plan.append((Tool.ROAD, cx1 + dx, cy1 + 8))
    # Wire to power second row
    for dx in range(-10, 18):
        plan.append((Tool.WIRE, cx1 + dx, cy1 + 9))
    # Third residential row
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 + 12))
    
    # Neighborhood 2: center area
    cx2, cy2 = 65, 45
    plan.append((Tool.COALPOWER, cx2 - 14, cy2))
    # Wire spine
    for dx in range(-12, 20):
        plan.append((Tool.WIRE, cx2 + dx, cy2))
    # Road below wire
    for dx in range(-12, 20):
        plan.append((Tool.ROAD, cx2 + dx, cy2 + 1))
    # Residential above
    for dx in range(-11, 20, 3):
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 - 3))
    # Mixed below road
    for i, dx in enumerate(range(-11, 20, 3)):
        if i % 3 == 0:
            plan.append((Tool.COMMERCIAL, cx2 + dx, cy2 + 4))
        elif i % 3 == 1:
            plan.append((Tool.INDUSTRIAL, cx2 + dx, cy2 + 4))
        else:
            plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 + 4))
    # Second row
    for dx in range(-11, 20, 3):
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 + 7))
    for dx in range(-12, 20):
        plan.append((Tool.ROAD, cx2 + dx, cy2 + 8))
    for dx in range(-12, 20):
        plan.append((Tool.WIRE, cx2 + dx, cy2 + 9))
    for i, dx in enumerate(range(-11, 20, 3)):
        if i % 2 == 0:
            plan.append((Tool.COMMERCIAL, cx2 + dx, cy2 + 12))
        else:
            plan.append((Tool.INDUSTRIAL, cx2 + dx, cy2 + 12))
    
    # Neighborhood 3: bottom-right
    cx3, cy3 = 85, 72
    plan.append((Tool.COALPOWER, cx3 - 12, cy3))
    for dx in range(-10, 18):
        plan.append((Tool.WIRE, cx3 + dx, cy3))
    for dx in range(-10, 18):
        plan.append((Tool.ROAD, cx3 + dx, cy3 + 1))
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 - 3))
    for i, dx in enumerate(range(-9, 18, 3)):
        if i % 2 == 0:
            plan.append((Tool.COMMERCIAL, cx3 + dx, cy3 + 4))
        else:
            plan.append((Tool.INDUSTRIAL, cx3 + dx, cy3 + 4))
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 + 7))
    for dx in range(-10, 18):
        plan.append((Tool.ROAD, cx3 + dx, cy3 + 8))
    for dx in range(-10, 18):
        plan.append((Tool.WIRE, cx3 + dx, cy3 + 9))
    for dx in range(-9, 18, 3):
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 + 12))
    
    # Neighborhood 4: top-right
    cx4, cy4 = 95, 15
    plan.append((Tool.COALPOWER, cx4 - 8, cy4))
    for dx in range(-6, 12):
        plan.append((Tool.WIRE, cx4 + dx, cy4))
    for dx in range(-6, 12):
        plan.append((Tool.ROAD, cx4 + dx, cy4 + 1))
    for dx in range(-5, 12, 3):
        plan.append((Tool.RESIDENTIAL, cx4 + dx, cy4 - 3))
    for i, dx in enumerate(range(-5, 12, 3)):
        if i % 2 == 0:
            plan.append((Tool.COMMERCIAL, cx4 + dx, cy4 + 4))
        else:
            plan.append((Tool.INDUSTRIAL, cx4 + dx, cy4 + 4))
    for dx in range(-5, 12, 3):
        plan.append((Tool.RESIDENTIAL, cx4 + dx, cy4 + 7))
    
    # Connect neighborhoods with vertical wire+road corridors
    # Corridor connecting N1 to N2
    conn_x1 = 40
    for y in range(cy1 + 1, cy2):
        plan.append((Tool.WIRE, conn_x1, y))
        plan.append((Tool.ROAD, conn_x1 + 1, y))
    
    # Corridor connecting N2 to N3
    conn_x2 = 75
    for y in range(cy2 + 1, cy3):
        plan.append((Tool.WIRE, conn_x2, y))
        plan.append((Tool.ROAD, conn_x2 + 1, y))
    
    # Wire connection N1 to N4 horizontally
    for x in range(cx1 + 15, cx4 - 6):
        plan.append((Tool.WIRE, x, cy1))
    
    # Re-place power plants to ensure they weren't overwritten
    plan.append((Tool.COALPOWER, cx1 - 12, cy1))
    plan.append((Tool.COALPOWER, cx2 - 14, cy2))
    plan.append((Tool.COALPOWER, cx3 - 12, cy3))
    plan.append((Tool.COALPOWER, cx4 - 8, cy4))
    
    # Add more residential zones to maximize population
    # Small dense cluster in the middle of map
    cx5, cy5 = 35, 60
    plan.append((Tool.COALPOWER, cx5 - 8, cy5))
    for dx in range(-6, 12):
        plan.append((Tool.WIRE, cx5 + dx, cy5))
    for dx in range(-6, 12):
        plan.append((Tool.ROAD, cx5 + dx, cy5 + 1))
    for dx in range(-5, 12, 3):
        plan.append((Tool.RESIDENTIAL, cx5 + dx, cy5 - 3))
    for i, dx in enumerate(range(-5, 12, 3)):
        if i % 2 == 0:
            plan.append((Tool.COMMERCIAL, cx5 + dx, cy5 + 4))
        else:
            plan.append((Tool.RESIDENTIAL, cx5 + dx, cy5 + 4))
    for dx in range(-5, 12, 3):
        plan.append((Tool.RESIDENTIAL, cx5 + dx, cy5 + 7))
    # Wire to connect N5 to N2
    for y in range(cy5 + 1, cy2):
        plan.append((Tool.WIRE, cx5 + 5, y))
    plan.append((Tool.COALPOWER, cx5 - 8, cy5))
    
    return plan


def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    
    i = state["i"]
    plan = state["plan"]
    
    if i >= len(plan):
        return None
    
    state["i"] = i + 1
    tool, x, y = plan[i]
    x = max(0, min(WORLD_W - 1, x))
    y = max(0, min(WORLD_H - 1, y))
    return (tool, x, y)