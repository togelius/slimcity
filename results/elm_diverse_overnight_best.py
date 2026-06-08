# cityPop=4460 fitness=4516.3 measures=(np.float64(0.502), np.float64(0.218)) origin=crossover
# replay-verified mean cityPop=4085 over 8 seeds (min 2740, max 4820)
def act(obs, state):
    """
    Combined policy:
    - Three neighborhoods (left, center, right) like Parent B
    - Each neighborhood has a COAL power plant (cheaper than nuclear, faster to place)
    - Horizontal wire spine + road spine (like Parent A's proven layout)
    - Zones above and below the spine (dense, mixed RES/COM/IND)
    - Additional zone rows for more population
    - Connecting roads between neighborhoods
    - Fire/police stations distributed
    """
    if "plan" not in state:
        plan = []

        # Three neighborhood centers
        neighborhoods = [
            (20, 35),   # left
            (60, 35),   # center  
            (100, 65),  # right (offset vertically for variety)
        ]

        zone_cycle = [Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL,
                      Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL]

        for idx, (cx, cy) in enumerate(neighborhoods):
            # Coal power plant to the left of the wire spine
            plant_x = cx - 2
            plant_y = cy
            plan.append((Tool.COALPOWER, plant_x, plant_y))

            # Horizontal wire spine at cy, road at cy+1
            # Wire from plant outward to the right
            for dx in range(-1, 22):
                wx = cx + dx
                if 0 <= wx <= 119:
                    plan.append((Tool.WIRE, wx, cy))
                rx = cx + dx
                if 0 <= rx <= 119:
                    plan.append((Tool.ROAD, rx, cy + 1))

            # Also road at cy-1 for zones above
            for dx in range(-1, 22):
                rx = cx + dx
                if 0 <= rx <= 119:
                    plan.append((Tool.ROAD, rx, cy - 1))

            # Zones above the spine (center at cy-3, then cy-6, cy-9)
            for row_offset in [-3, -6, -9, -12]:
                zone_y = cy + row_offset
                if zone_y < 3 or zone_y > 96:
                    continue
                k = 0
                for dx in range(0, 21, 3):
                    zx = cx + dx
                    if zx > 119 or zx < 0:
                        continue
                    zt = zone_cycle[(k + idx) % len(zone_cycle)]
                    plan.append((zt, zx, zone_y))
                    k += 1

            # Zones below the spine (center at cy+4, then cy+7, cy+10)
            for row_offset in [4, 7, 10, 13]:
                zone_y = cy + row_offset
                if zone_y < 3 or zone_y > 96:
                    continue
                k = 0
                for dx in range(0, 21, 3):
                    zx = cx + dx
                    if zx > 119 or zx < 0:
                        continue
                    zt = zone_cycle[(k + idx + 2) % len(zone_cycle)]
                    plan.append((zt, zx, zone_y))
                    k += 1

        # Horizontal connecting road at y=50 and y=51
        for x in range(0, 120):
            plan.append((Tool.ROAD, x, 50))

        # Horizontal wire at y=49 connecting all power plants
        for x in range(0, 120):
            plan.append((Tool.WIRE, x, 49))

        # Vertical wires from each plant down/up to the horizontal wire spine
        # Left neighborhood: plant at (18, 35), wire spine at y=49
        plan.append((Tool.COALPOWER, 18, 35))
        for y in range(35, 50):
            plan.append((Tool.WIRE, 18, y))

        # Center neighborhood: plant at (58, 35)
        plan.append((Tool.COALPOWER, 58, 35))
        for y in range(35, 50):
            plan.append((Tool.WIRE, 58, y))

        # Right neighborhood: plant at (98, 65), wire spine at y=49
        plan.append((Tool.COALPOWER, 98, 65))
        for y in range(49, 66):
            plan.append((Tool.WIRE, 98, y))

        # Additional zone cluster around y=65 for left/center
        for cx, cy in [(20, 65), (60, 65)]:
            plan.append((Tool.COALPOWER, cx - 2, cy))
            for dx in range(-1, 22):
                wx = cx + dx
                if 0 <= wx <= 119:
                    plan.append((Tool.WIRE, wx, cy))
                rx = cx + dx
                if 0 <= rx <= 119:
                    plan.append((Tool.ROAD, rx, cy + 1))
                    plan.append((Tool.ROAD, rx, cy - 1))
            # wire from this plant to main wire spine
            for y in range(50, cy):
                plan.append((Tool.WIRE, cx - 2, y))

            for row_offset in [-3, -6, 4, 7]:
                zone_y = cy + row_offset
                if zone_y < 3 or zone_y > 96:
                    continue
                for dx in range(0, 21, 3):
                    zx = cx + dx
                    if zx > 119 or zx < 0:
                        continue
                    k = dx // 3
                    zt = zone_cycle[(k + 1) % len(zone_cycle)]
                    plan.append((zt, zx, zone_y))

        # Services distributed
        plan.append((Tool.FIRESTATION, 10, 20))
        plan.append((Tool.POLICESTATION, 30, 20))
        plan.append((Tool.FIRESTATION, 55, 20))
        plan.append((Tool.POLICESTATION, 75, 20))
        plan.append((Tool.FIRESTATION, 105, 20))
        plan.append((Tool.POLICESTATION, 10, 80))
        plan.append((Tool.FIRESTATION, 55, 80))
        plan.append((Tool.POLICESTATION, 105, 80))

        # Stadium for happiness boost
        plan.append((Tool.STADIUM, 55, 90))

        state["plan"] = plan
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
