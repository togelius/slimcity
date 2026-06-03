# it61 pop=2440
def act(obs, state):
    if "plan" not in state:
        plan = []
        
        # === NEIGHBORHOOD 1: Top-left area, industrial/commercial focus ===
        cx1, cy1 = 30, 30
        
        # Coal power plant
        plan.append((Tool.COALPOWER, cx1, cy1 - 8))
        
        # Wire from plant down to main road level
        for dy in range(-7, 1):
            plan.append((Tool.WIRE, cx1, cy1 + dy))
        
        # Horizontal wire spine
        for dx in range(-8, 9):
            plan.append((Tool.WIRE, cx1 + dx, cy1))
        
        # Horizontal road just below wire spine
        for dx in range(-10, 11):
            plan.append((Tool.ROAD, cx1 + dx, cy1 + 1))
        
        # Vertical roads for grid
        for dy in range(-2, 12):
            plan.append((Tool.ROAD, cx1 - 7, cy1 + dy))
            plan.append((Tool.ROAD, cx1 + 7, cy1 + dy))
        
        # Industrial zones (north of road)
        for dx in [-6, -3, 0, 3, 6]:
            plan.append((Tool.INDUSTRIAL, cx1 + dx, cy1 - 3))
        
        # Commercial zones (south of road, left side)
        for dy in [4, 7, 10]:
            plan.append((Tool.COMMERCIAL, cx1 - 4, cy1 + dy))
            plan.append((Tool.COMMERCIAL, cx1 - 1, cy1 + dy))
        
        # Residential zones (south of road, right side)
        for dy in [4, 7, 10]:
            plan.append((Tool.RESIDENTIAL, cx1 + 2, cy1 + dy))
            plan.append((Tool.RESIDENTIAL, cx1 + 5, cy1 + dy))
        
        # Wire to south zones
        for dy in range(1, 12):
            plan.append((Tool.WIRE, cx1 - 4, cy1 + dy))
            plan.append((Tool.WIRE, cx1 + 5, cy1 + dy))
        
        # Fire and police
        plan.append((Tool.FIRESTATION, cx1 + 9, cy1 - 3))
        plan.append((Tool.POLICESTATION, cx1 - 9, cy1 - 3))
        
        # === NEIGHBORHOOD 2: Center, residential focus ===
        cx2, cy2 = 65, 55
        
        # Nuclear power plant
        plan.append((Tool.NUCLEARPOWER, cx2 + 10, cy2))
        
        # Wire spine horizontal
        for dx in range(-10, 11):
            plan.append((Tool.WIRE, cx2 + dx, cy2))
        
        # Main horizontal roads above and below wire
        for dx in range(-12, 13):
            plan.append((Tool.ROAD, cx2 + dx, cy2 - 1))
            plan.append((Tool.ROAD, cx2 + dx, cy2 + 1))
        
        # Vertical roads
        for dy in range(-10, 11):
            plan.append((Tool.ROAD, cx2 - 9, cy2 + dy))
            plan.append((Tool.ROAD, cx2 - 3, cy2 + dy))
            plan.append((Tool.ROAD, cx2 + 3, cy2 + dy))
            plan.append((Tool.ROAD, cx2 + 9, cy2 + dy))
        
        # Wire vertically to reach all zones
        for dy in range(-10, 11):
            plan.append((Tool.WIRE, cx2 - 6, cy2 + dy))
            plan.append((Tool.WIRE, cx2 + 6, cy2 + dy))
        
        # Residential zones - dense grid
        for dy in [-7, -4, 4, 7]:
            for dx in [-7, -4, -1]:
                plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 + dy))
        
        # Commercial zones
        for dy in [-7, -4, 4, 7]:
            for dx in [1, 4, 7]:
                plan.append((Tool.COMMERCIAL, cx2 + dx, cy2 + dy))
        
        # Industrial zones at edges
        for dx in [-11, 11]:
            for dy in [-7, -4, -1, 2, 5]:
                plan.append((Tool.INDUSTRIAL, cx2 + dx, cy2 + dy))
        
        # Fire and police
        plan.append((Tool.FIRESTATION, cx2 - 11, cy2 - 10))
        plan.append((Tool.POLICESTATION, cx2 + 11, cy2 + 10))
        
        # === NEIGHBORHOOD 3: Bottom-right, mixed ===
        cx3, cy3 = 95, 75
        
        # Coal power plant
        plan.append((Tool.COALPOWER, cx3 - 8, cy3 + 5))
        
        # Wire
        for dx in range(-8, 6):
            plan.append((Tool.WIRE, cx3 + dx, cy3))
        for dy in range(-8, 6):
            plan.append((Tool.WIRE, cx3, cy3 + dy))
        
        # Roads grid
        for dy in range(-8, 9):
            plan.append((Tool.ROAD, cx3 - 5, cy3 + dy))
            plan.append((Tool.ROAD, cx3 + 5, cy3 + dy))
        for dx in range(-8, 9):
            plan.append((Tool.ROAD, cx3 + dx, cy3 - 5))
            plan.append((Tool.ROAD, cx3 + dx, cy3 + 5))
        
        # Zones
        for dy in [-8, -5, -2, 1]:
            plan.append((Tool.RESIDENTIAL, cx3 - 3, cy3 + dy))
            plan.append((Tool.COMMERCIAL, cx3 + 3, cy3 + dy))
        
        plan.append((Tool.FIRESTATION, cx3, cy3 - 8))
        
        state["plan"] = plan
        state["i"] = 0
    
    i = state["i"]
    plan = state["plan"]
    
    if i < len(plan):
        state["i"] = i + 1
        return plan[i]
    
    # Fill remaining steps with more residential zones around neighborhood 2
    if "fill_i" not in state:
        fill = []
        for ey in range(15, 45, 6):
            for ex in range(10, 115, 6):
                fill.append((Tool.ROAD, ex, ey))
                fill.append((Tool.WIRE, ex + 3, ey))
                fill.append((Tool.RESIDENTIAL, ex + 1, ey - 2))
                fill.append((Tool.COMMERCIAL, ex + 4, ey - 2))
        state["fill"] = fill
        state["fill_i"] = 0
    
    fi = state["fill_i"]
    if fi < len(state["fill"]):
        state["fill_i"] = fi + 1
        return state["fill"][fi]
    
    return None