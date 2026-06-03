# fitness=2690.8 cityPop=2668 measures=(0.4494409937888199, 0.2134782608695652) origin=mutate
def act(obs, state):
    if "plan" not in state:
        plan = []

        # Strategy: Horizontal spine layout instead of vertical.
        # Each neighborhood has a horizontal wire/road pair with zones above and below.
        # More commercial/industrial balance, denser zone packing.
        # Two rows of horizontal neighborhoods.

        def add_horiz_neighborhood(plan, cx, cy, width=30):
            """
            Horizontal spine: wire on row cy, road on row cy+1.
            Zones above (centers at cy-2) and below (centers at cy+3).
            Power plant at left end of wire.
            """
            # Power plant at start
            plan.append((Tool.COALPOWER, cx, cy))

            # Wire runs right from plant
            for dx in range(0, width):
                plan.append((Tool.WIRE, cx + dx, cy))

            # Road runs parallel below wire
            for dx in range(0, width):
                plan.append((Tool.ROAD, cx + dx, cy + 1))

            # Zones above wire (center row = cy-2, so zone spans cy-4 to cy)
            # Place every 3 columns
            zone_types_above = [Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL,
                                 Tool.RESIDENTIAL, Tool.INDUSTRIAL, Tool.COMMERCIAL,
                                 Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL]
            for i, dx in enumerate(range(1, width - 1, 3)):
                zt = zone_types_above[i % len(zone_types_above)]
                plan.append((zt, cx + dx, cy - 2))

            # Zones below road (center row = cy+3)
            zone_types_below = [Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.RESIDENTIAL,
                                 Tool.INDUSTRIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL,
                                 Tool.RESIDENTIAL, Tool.INDUSTRIAL, Tool.RESIDENTIAL, Tool.COMMERCIAL]
            for i, dx in enumerate(range(1, width - 1, 3)):
                zt = zone_types_below[i % len(zone_types_below)]
                plan.append((zt, cx + dx, cy + 3))

            # Fire and police stations
            plan.append((Tool.FIRESTATION, cx + width - 2, cy - 2))
            plan.append((Tool.POLICESTATION, cx + width - 2, cy + 3))

        # === Row 1: Three neighborhoods side by side ===
        # Neighborhood A
        add_horiz_neighborhood(plan, cx=5, cy=20, width=32)
        # Neighborhood B
        add_horiz_neighborhood(plan, cx=42, cy=20, width=32)
        # Neighborhood C
        add_horiz_neighborhood(plan, cx=79, cy=20, width=32)

        # Connect row 1 neighborhoods with vertical roads at their junctions
        for y in range(15, 26):
            plan.append((Tool.ROAD, 41, y))
            plan.append((Tool.ROAD, 78, y))

        # Horizontal road connecting all three at road level
        for x in range(5, 111):
            plan.append((Tool.ROAD, x, 21))

        # Wire connecting all three at wire level
        for x in range(5, 111):
            plan.append((Tool.WIRE, x, 20))

        # === Row 2: Three neighborhoods, offset ===
        add_horiz_neighborhood(plan, cx=5, cy=55, width=32)
        add_horiz_neighborhood(plan, cx=42, cy=55, width=32)
        add_horiz_neighborhood(plan, cx=79, cy=55, width=32)

        # Connect row 2
        for x in range(5, 111):
            plan.append((Tool.ROAD, x, 56))
        for x in range(5, 111):
            plan.append((Tool.WIRE, x, 55))

        # === Row 3: Two neighborhoods for bottom strip ===
        add_horiz_neighborhood(plan, cx=15, cy=82, width=36)
        add_horiz_neighborhood(plan, cx=65, cy=82, width=36)

        for x in range(15, 101):
            plan.append((Tool.ROAD, x, 83))
        for x in range(15, 101):
            plan.append((Tool.WIRE, x, 82))

        # === Vertical connectors between rows ===
        # Connect row1 to row2 with vertical roads
        for x in [21, 58, 95]:
            for y in range(23, 53):
                plan.append((Tool.ROAD, x, y))
            # Wire alongside
            for y in range(22, 54):
                plan.append((Tool.WIRE, x + 1, y))

        # Connect row2 to row3
        for x in [33, 83]:
            for y in range(58, 80):
                plan.append((Tool.ROAD, x, y))
            for y in range(57, 81):
                plan.append((Tool.WIRE, x + 1, y))

        # Additional nuclear power for extra capacity
        plan.append((Tool.NUCLEARPOWER, 110, 20))
        plan.append((Tool.WIRE, 110, 20))
        plan.append((Tool.NUCLEARPOWER, 110, 55))
        plan.append((Tool.WIRE, 110, 55))

        # Extra zones in gaps
        # Fill some empty space between row1 and row2 with more zones
        extra_positions = [
            (Tool.RESIDENTIAL, 10, 38), (Tool.COMMERCIAL, 25, 38),
            (Tool.INDUSTRIAL, 40, 38), (Tool.RESIDENTIAL, 55, 38),
            (Tool.COMMERCIAL, 70, 38), (Tool.RESIDENTIAL, 85, 38),
            (Tool.INDUSTRIAL, 100, 38),
        ]
        for item in extra_positions:
            plan.append(item)
            # Wire and road nearby
            plan.append((Tool.WIRE, item[1], item[2] + 2))
            plan.append((Tool.ROAD, item[1], item[2] + 3))

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
