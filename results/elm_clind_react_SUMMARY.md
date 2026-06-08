# SlimCity ELM run — elm_clind_react

Quality-Diversity (MAP-Elites) over **Python-code genomes**, with **Claude (Sonnet) as the mutation/crossover operator**. Each genome is a closed-loop `act(obs, state)` tile-placement policy; fitness = cityPop growth (dense-shaped), behavior = (residential share, industrial share); fitness is the mean of 5 independent rolls (the engine eval is ~19% noisy per process).

## Headline
- **Best city: cityPop 5380** (robust mean), fitness 5400, found at iter 5690 via **crossover**, behavior (cf_sensitivity=0.00, traj_divergence=0.00, ind_share=0.00).
- For comparison, the prior CMA-ME state of the art was cityPop 1,680 (layout genome) / 1,120 (action tape). **ELM beats it.**
- Archive: **316 cells filled** (49.4% of 640), QD-score **266771**.
- Iterations completed: **6000**. Genomes that grew a city (pop>0): **2792/5976** evaluated.

## Operator efficiency
- eval=5976  op_error=7  invalid=0  (of 5983 attempts)
- operator failure rate: 0% op-error, 0% invalid.

## Top cities (robust mean over 5 rolls)

| rank | cityPop | fitness | cf_sensitivity | traj_divergence | ind_share | origin | iter |
|---|---|---|---|---|---|---|---|
| 1 | 5380 | 5400 | 0.00 | 0.00 | 0.00 | crossover | 5690 |
| 2 | 4500 | 4517 | 1.00 | 0.02 | 0.21 | crossover | 5901 |
| 3 | 3227 | 3240 | 1.00 | 0.22 | 0.20 | mutate+cl | 5514 |
| 4 | 3113 | 3128 | 0.53 | 0.06 | 0.13 | mutate+cl | 4002 |
| 5 | 2293 | 2311 | 1.00 | 0.02 | 0.19 | mutate+cl | 1481 |
| 6 | 2187 | 2198 | 0.73 | 0.27 | 0.26 | crossover | 603 |
| 7 | 2187 | 2196 | 0.39 | 0.00 | 0.15 | crossover | 4491 |
| 8 | 2053 | 2068 | 0.99 | 0.43 | 0.25 | crossover+cl | 3358 |
| 9 | 1973 | 1982 | 0.44 | 0.62 | 0.21 | mutate | 4044 |
| 10 | 1960 | 1979 | 0.95 | 0.64 | 0.17 | crossover | 4321 |

## QD-score / coverage progress

| iter | filled | coverage | qd_score | best_pop |
|---|---|---|---|---|
| 0 | 2 | 0.003 | 545 | 313 |
| 502 | 39 | 0.061 | 10872 | 873 |
| 1002 | 94 | 0.147 | 56270 | 2187 |
| 1505 | 141 | 0.220 | 96906 | 2293 |
| 2006 | 177 | 0.277 | 130692 | 3100 |
| 2506 | 204 | 0.319 | 155548 | 3100 |
| 3004 | 232 | 0.362 | 182475 | 3100 |
| 3508 | 250 | 0.391 | 201036 | 4327 |
| 4008 | 266 | 0.416 | 218620 | 4327 |
| 4508 | 286 | 0.447 | 235882 | 4327 |
| 5009 | 299 | 0.467 | 245715 | 4327 |
| 5508 | 304 | 0.475 | 253564 | 4327 |
| 6000 | 316 | 0.494 | 266771 | 5380 |

## Origin of elites
- crossover=68, seed=1, mutate+cl=110, mutate=86, crossover+cl=51

## Champion policy (source)
```python
def act(obs, state):
    """
    Combined policy:
    - Parent A's structured multi-cluster pre-planned layout
    - Parent B's reactive closed-loop expansion and zone placement
    - Better power connectivity and zone placement logic
    """
    
    def build_plan():
        plan = []
        
        # === CLUSTER 1: Left-center at (30, 40) ===
        cx1, cy1 = 30, 40
        # Coal power plant above cluster
        plan.append((Tool.COALPOWER, cx1, cy1 - 12))
        # Wire from plant down to road spine
        for dy in range(-11, 1):
            plan.append((Tool.WIRE, cx1, cy1 + dy))
        # Horizontal road spine
        for dx in range(-8, 9):
            plan.append((Tool.ROAD, cx1 + dx, cy1))
        # Wire parallel to road for power distribution
        for dx in range(-8, 9):
            plan.append((Tool.WIRE, cx1 + dx, cy1 - 1))
            plan.append((Tool.WIRE, cx1 + dx, cy1 + 1))
        # Vertical road
        for dy in range(-8, 9):
            plan.append((Tool.ROAD, cx1, cy1 + dy))
        # Residential zones
        for dy in [-5, -2, 2, 5]:
            for dx in [-5, -2, 2, 5]:
                plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 + dy))
        # Commercial zones
        for dy in [-5, 5]:
            plan.append((Tool.COMMERCIAL, cx1 - 5, cy1 + dy))
            plan.append((Tool.COMMERCIAL, cx1 + 5, cy1 + dy))
        # Industrial south
        for dx in [-5, -2, 0, 2, 5]:
            plan.append((Tool.INDUSTRIAL, cx1 + dx, cy1 + 7))
        # Fire/Police
        plan.append((Tool.FIRESTATION, cx1 + 10, cy1 - 3))
        plan.append((Tool.POLICESTATION, cx1 - 10, cy1 - 3))
        
        # === CLUSTER 2: Right-center at (85, 40) ===
        cx2, cy2 = 85, 40
        plan.append((Tool.COALPOWER, cx2, cy2 - 12))
        for dy in range(-11, 1):
            plan.append((Tool.WIRE, cx2, cy2 + dy))
        for dx in range(-8, 9):
            plan.append((Tool.ROAD, cx2 + dx, cy2))
        for dx in range(-8, 9):
            plan.append((Tool.WIRE, cx2 + dx, cy2 - 1))
            plan.append((Tool.WIRE, cx2 + dx, cy2 + 1))
        for dy in range(-8, 9):
            plan.append((Tool.ROAD, cx2, cy2 + dy))
        for dy in [-5, -2, 2, 5]:
            plan.append((Tool.COMMERCIAL, cx2 - 5, cy2 + dy))
            plan.append((Tool.COMMERCIAL, cx2 + 5, cy2 + dy))
            plan.append((Tool.RESIDENTIAL, cx2 - 2, cy2 + dy))
            plan.append((Tool.RESIDENTIAL, cx2 + 2, cy2 + dy))
        for dx in [-5, -2, 0, 2, 5]:
            plan.append((Tool.INDUSTRIAL, cx2 + dx, cy2 + 7))
        plan.append((Tool.FIRESTATION, cx2 + 10, cy2 - 3))
        plan.append((Tool.POLICESTATION, cx2 - 10, cy2 - 3))
        
        # === CLUSTER 3: Center-bottom at (57, 70) ===
        cx3, cy3 = 57, 70
        plan.append((Tool.COALPOWER, cx3, cy3 - 12))
        for dy in range(-11, 1):
            plan.append((Tool.WIRE, cx3, cy3 + dy))
        for dx in range(-8, 9):
            plan.append((Tool.ROAD, cx3 + dx, cy3))
        for dx in range(-8, 9):
            plan.append((Tool.WIRE, cx3 + dx, cy3 - 1))
            plan.append((Tool.WIRE, cx3 + dx, cy3 + 1))
        for dy in range(-8, 9):
            plan.append((Tool.ROAD, cx3, cy3 + dy))
        for dy in [-5, -2, 2, 5]:
            for dx in [-5, -2, 2, 5]:
                plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 + dy))
        for dx in [-5, -2, 0, 2, 5]:
            plan.append((Tool.COMMERCIAL, cx3 + dx, cy3 + 7))
        plan.append((Tool.FIRESTATION, cx3 + 10, cy3 - 3))
        plan.append((Tool.POLICESTATION, cx3 - 10, cy3 - 3))
        
        # === CONNECTING ROADS & WIRES between clusters ===
        # Horizontal connector top: cluster1 to cluster2
        for x in range(cx1 + 9, cx2 - 8):
            plan.append((Tool.ROAD, x, cy1))
            plan.append((Tool.WIRE, x, cy1 - 1))
        # Vertical connector: cluster1 to cluster3
        for y in range(cy1 + 9, cy3 - 8):
            plan.append((Tool.ROAD, cx1, y))
            plan.append((Tool.WIRE, cx1 + 1, y))
        # Vertical connector: cluster2 to cluster3
        for y in range(cy2 + 9, cy3 - 8):
            plan.append((Tool.ROAD, cx2, y))
            plan.append((Tool.WIRE, cx2 + 1, y))
        # Horizontal connector bottom: cluster1 to cluster2 at cy3
        for x in range(cx1 + 9, cx2 - 8):
            plan.append((Tool.ROAD, x, cy3))
            plan.append((Tool.WIRE, x, cy3 - 1))
        
        # Stadium and seaport for bonus
        plan.append((Tool.STADIUM, 57, 20))
        
        return plan
    
    # Initialize state
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
        state["phase"] = "plan"
    
    i = state["i"]
    plan = state["plan"]
    
    # Execute pre-built plan first
    if i < len(plan):
        state["i"] = i + 1
        tool, x, y = plan[i]
        x = max(0, min(WORLD_W - 1, x))
        y = max(0, min(WORLD_H - 1, y))
        return (tool, x, y)
    
    # === REACTIVE EXPANSION PHASE (from Parent B) ===
    tm = obs.tile_map
    em = empty_mask(tm)
    wm = wire_mask(tm)
    rm = road_mask(tm)
    pm = plant_mask(tm)
    resm = res_mask(tm)
    comm = com_mask(tm)
    indm = ind_mask(tm)
    
    pop = obs.city_pop
    step = obs.step
    powered = obs.powered_zones
    
    resm_c = int(np.sum(resm))
    comm_c = int(np.sum(comm))
    indm_c = int(np.sum(indm))
    zone_count = resm_c + comm_c + indm_c
    wire_count = int(np.sum(wm))
    road_count = int(np.sum(rm))
    
    CX, CY = 57, 55
    
    def in_bounds(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H
    
    power_map = pm | wm
    
    # Build neighbor masks for road and power
    def make_nbr(base_mask, radius):
        nbr = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) <= radius:
                    sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
                    sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
                    r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
                    c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
                    nbr[r0:r1, c0:c1] |= base_mask[sr0:sr1, sc0:sc1]
        return nbr
    
    nbr_road = make_nbr(rm, 2)
    nbr_power = make_nbr(power_map, 3)
    
    zone_existing = resm | comm | indm
    
    def zone_clear(zx, zy):
        if zy < 2 or zy > WORLD_H - 3 or zx < 2 or zx > WORLD_W - 3:
            return False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = zy + dr, zx + dc
                if not in_bounds(nc, nr):
                    return False
                if not (bool(em[nr, nc]) or bool(zone_existing[nr, nc])):
                    return False
        return True
    
    def pick_tool():
        total = max(resm_c + comm_c + indm_c, 1)
        r = resm_c / total
        c = comm_c / total
        ind = indm_c / total
        if ind < 0.20:
            return Tool.INDUSTRIAL
        if c < 0.30:
            return Tool.COMMERCIAL
        if r < 0.50:
            return Tool.RESIDENTIAL
        return Tool.RESIDENTIAL
    
    # Place zone where road+power access exists
    def place_zone():
        cand = em & nbr_road & nbr_power
        cand[:3, :] = False; cand[WORLD_H-3:, :] = False
        cand[:, :3] = False; cand[:, WORLD_W-3:] = False
        
        cys_arr, cxs_arr = np.where(cand)
        if len(cxs_arr) == 0:
            return None
        
        scores = np.zeros(len(cxs_arr), dtype=float)
        d = np.abs(cxs_arr - CX) + np.abs(cys_arr - CY)
        scores -= d * 0.1
        
        # Bonus for direct road adjacency
        for k in range(len(cxs_arr)):
            zx, zy = int(cxs_arr[k]), int(cys_arr[k])
            for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx2, ny2 = zx+dx2, zy+dy2
                if in_bounds(nx2, ny2) and bool(rm[ny2, nx2]):
                    scores[k] += 10
            for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx2, ny2 = zx+dx2, zy+dy2
                if in_bounds(nx2, ny2) and bool(power_map[ny2, nx2]):
                    scores[k] += 5
        
        tool = pick_tool()
        order = np.argsort(-scores)
        for idx in range(min(len(order), 400)):
            k = order[idx]
            zx, zy = int(cxs_arr[k]), int(cys_arr[k])
            if zone_clear(zx, zy):
                return (tool, zx, zy)
        return None
    
    # Extend wire toward unpowered zones
    def extend_wire():
        py_arr, px_arr = np.where(power_map)
        if len(px_arr) == 0:
            return None
        zone_map = resm | comm | indm
        best, best_score = None, -1e9
        step_s = max(1, len(px_arr) // 300)
        for idx in range(0, len(px_arr), step_s):
            px2, py2 = int(px_arr[idx]), int(py_arr[idx])
            d_center = abs(px2 - CX) + abs(py2 - CY)
            if d_center > 60:
                continue
            for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
                wx, wy = px2+ddx, py2+ddy
                if not in_bounds(wx, wy): continue
                if bool(pm[wy, wx]) or bool(wm[wy, wx]): continue
                if bool(rm[wy, wx]): continue
                if not bool(em[wy, wx]): continue
                score = 0
                d = abs(wx - CX) + abs(wy - CY)
                score -= d * 0.2
                for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)]:
                    nx2, ny2 = wx+dx2, wy+dy2
                    if in_bounds(nx2, ny2) and bool(zone_map[ny2, nx2]):
                        has_power = False
                        for dx3, dy3 in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx3, ny3 = nx2+dx3, ny2+dy3
                            if in_bounds(nx3, ny3) and bool(power_map[ny3, nx3]):
                                has_power = True; break
                        score += 15 if not has_power else 3
                for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx2, ny2 = wx+dx2, wy+dy2
                    if in_bounds(nx2, ny2) and bool(rm[ny2, nx2]):
                        score += 5
                if score > best_score:
                    best_score = score
                    best = (Tool.WIRE, wx, wy)
        return best
    
    # Extend road
    def extend_road():
        ry_arr, rx_arr = np.where(rm)
        if len(rx_arr) == 0:
            return (Tool.ROAD, CX, CY + 4)
        best, best_score = None, -1e9
        step_s = max(1, len(rx_arr) // 300)
        for idx in range(0, len(rx_arr), step_s):
            rx2, ry2 = int(rx_arr[idx]), int(ry_arr[idx])
            d_center = abs(rx2 - CX) + abs(ry2 - CY)
            if d_center > 55:
                continue
            for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx2, ny2 = rx2+ddx, ry2+ddy
                if not in_bounds(nx2, ny2): continue
                if bool(rm[ny2, nx2]) or bool(pm[ny2, nx2]): continue
                if not bool(em[ny2, nx2]): continue
                score = 0
                d = abs(nx2 - CX) + abs(ny2 - CY)
                score -= d * 0.1
                for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)]:
                    nx3, ny3 = nx2+dx2, ny2+dy2
                    if in_bounds(nx3, ny3):
                        if bool(resm[ny3,nx3]) or bool(comm[ny3,nx3]) or bool(indm[ny3,nx3]):
                            score += 8
                for dx2, dy2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx3, ny3 = nx2+dx2, ny2+dy2
                    if in_bounds(nx3, ny3) and bool(power_map[ny3, nx3]):
                        score += 4
                if score > best_score:
                    best_score = score
                    best = (Tool.ROAD, nx2, ny2)
        return best
    
    # Services
    def try_services():
        clusters = [(30, 40), (85, 40), (57, 70)]
        if pop > 100:
            for cx_s, cy_s in clusters:
                for fx, fy in [(cx_s+11, cy_s), (cx_s-11, cy_s), (cx_s, cy_s+11)]:
                    if in_bounds(fx, fy) and bool(em[fy, fx]):
                        return (Tool.FIRESTATION, fx, fy)
        if pop > 300:
            for cx_s, cy_s in clusters:
                for fx, fy in [(cx_s+11, cy_s+5), (cx_s-11, cy_s+5)]:
                    if in_bounds(fx, fy) and bool(em[fy, fx]):
                        return (Tool.POLICESTATION, fx, fy)
        return None
    
    # Additional power plants if needed
    plant_ys, plant_xs = np.where(pm)
    n_plants = len(plant_xs)
    if n_plants < 4 and pop > 1500:
        for tx, ty in [(57, 25), (15, 55), (100, 55), (57, 85)]:
            if in_bounds(tx, ty) and bool(em[ty, tx]):
                return (Tool.COALPOWER, tx, ty)
    
    wire_action = extend_wire()
    road_action = extend_road()
    
    svc = try_services()
    if svc:
        return svc
    
    unpowered_zones = max(zone_count - powered, 0)
    step_mod = step % 10
    
    if zone_count == 0:
        if wire_action: return wire_action
        if road_action: return road_action
        z = place_zone()
        if z: return z
        return wire_action
    
    elif unpowered_zones > zone_count * 0.3:
        if wire_action: return wire_action
        z = place_zone()
        if z: return z
        if road_action: return road_action
    
    elif pop < 500:
        if step_mod == 0 and wire_action: return wire_action
        if step_mod in (1, 2) and road_action: return road_action
        z = place_zone()
        if z: return z
        if road_action: return road_action
        if wire_action: return wire_action
    
    else:
        # Late game: maximize zones
        z = place_zone()
        if z: return z
        if step_mod < 2 and wire_action: return wire_action
        if step_mod < 4 and road_action: return road_action
        if wire_action: return wire_action
        if road_action: return road_action
    
    return None
```

Artifacts: `results/elm_clind_react.npz` (archive, every cell-winner's full source), `results/elm_clind_react_generations.jsonl` (every genome generated), `results/elm_clind_react_champion.py`.
