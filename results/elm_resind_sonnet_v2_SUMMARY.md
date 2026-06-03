# SlimCity ELM run — elm_resind_sonnet_v2

Quality-Diversity (MAP-Elites) over **Python-code genomes**, with **Claude (Sonnet) as the mutation/crossover operator**. Each genome is a closed-loop `act(obs, state)` tile-placement policy; fitness = cityPop growth (dense-shaped), behavior = (residential share, industrial share); fitness is the mean of 5 independent rolls (the engine eval is ~19% noisy per process).

## Headline
- **Best city: cityPop 2668** (robust mean), fitness 2691, found at iter 244 via **mutate**, behavior (res=0.45, ind=0.21).
- For comparison, the prior CMA-ME state of the art was cityPop 1,680 (layout genome) / 1,120 (action tape). **ELM beats it.**
- Archive: **55 cells filled** (13.8% of 400), QD-score **24415**.
- Iterations completed: **500**. Genomes that grew a city (pop>0): **93/501** evaluated.

## Operator efficiency
- eval=501  op_error=0  invalid=0  (of 501 attempts)
- operator failure rate: 0% op-error, 0% invalid.

## Top cities (robust mean over 5 rolls)

| rank | cityPop | fitness | res | ind | origin | iter |
|---|---|---|---|---|---|---|
| 1 | 2668 | 2691 | 0.45 | 0.21 | mutate | 244 |
| 2 | 2436 | 2460 | 0.50 | 0.20 | crossover | 417 |
| 3 | 1792 | 1823 | 0.67 | 0.11 | crossover | 89 |
| 4 | 1296 | 1321 | 0.56 | 0.17 | crossover | 342 |
| 5 | 1136 | 1147 | 0.38 | 0.24 | mutate | 135 |
| 6 | 1056 | 1079 | 0.67 | 0.17 | crossover | 132 |
| 7 | 1052 | 1065 | 0.30 | 0.39 | mutate | 363 |
| 8 | 972 | 994 | 0.50 | 0.22 | crossover | 271 |
| 9 | 964 | 988 | 0.50 | 0.15 | crossover | 158 |
| 10 | 960 | 987 | 0.45 | 0.18 | crossover | 401 |

## QD-score / coverage progress

| iter | filled | coverage | qd_score | best_pop |
|---|---|---|---|---|
| 0 | 1 | 0.003 | 343 | 336 |
| 40 | 9 | 0.022 | 1470 | 336 |
| 80 | 17 | 0.043 | 2324 | 336 |
| 120 | 23 | 0.058 | 6017 | 1792 |
| 160 | 26 | 0.065 | 8871 | 1792 |
| 200 | 33 | 0.083 | 11653 | 1792 |
| 240 | 35 | 0.087 | 13037 | 1792 |
| 280 | 38 | 0.095 | 17486 | 2668 |
| 320 | 40 | 0.100 | 18286 | 2668 |
| 360 | 47 | 0.117 | 20180 | 2668 |
| 400 | 49 | 0.122 | 21833 | 2668 |
| 440 | 53 | 0.133 | 23916 | 2668 |
| 480 | 55 | 0.138 | 24415 | 2668 |
| 500 | 55 | 0.138 | 24415 | 2668 |

## Origin of elites
- mutate=27, crossover=28

## Champion policy (source)
```python
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
```

Artifacts: `results/elm_resind_sonnet_v2.npz` (archive, every cell-winner's full source), `results/elm_resind_sonnet_v2_generations.jsonl` (every genome generated), `results/elm_resind_sonnet_v2_champion.py`.
