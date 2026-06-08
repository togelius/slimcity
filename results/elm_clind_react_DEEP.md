# Deep analysis — elm_clind_react
archive: 316 cells, qd=266771, best cityPop=5380

## champion (max cityPop)
- cityPop=4280 (R=276 C=0 I=0) | stored cityPop=5380 | origin=crossover iter=5690
- reactivity: cf_sensitivity=0.00, traj_divergence=0.00, ind_share=0.00  [reactive-map]
- code: 369 lines, 14033 chars | motifs: precomputed-plan, reads-map, reads-stats, plants=4c, wire, road
```
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
...............*..........................*.................
............................................................
...............=............................................
............RRRRRRR.........................................
............RRRRRRR.........................................
............RRRRRRR.........................................
...........=RRRRRRR=........................................
............RRRRRRR.........................................
............RRRRRRR.........................................
............RRRRRRR.........................................
...............=............................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
```
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

## best reactive
- cityPop=2720 (R=34 C=3 I=6) | stored cityPop=4500 | origin=crossover iter=5901
- reactivity: cf_sensitivity=1.00, traj_divergence=0.02, ind_share=0.21  [reactive-map]
- code: 316 lines, 10631 chars | motifs: reads-map, reads-stats, plants=3c, wire, road
```
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
..............................=.............................
..............................=.............................
............................=====...........................
............................=.=.=...........................
..........................CCIIRRCC=.........................
..........................CCIIRRCC=.........................
........................===CII..II==........................
........................=.=CII..II=.=.......................
......................====RR==*.RR===.......................
........................=.RRRR..RR=.........................
........................==RRRR..CC=.........................
..........................RR....CC..........................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
```
```python
def act(obs, state):
    """
    Hybrid: Single cluster (Parent B's compact approach) with Parent A's zone mixing.
    Key improvements:
    - Compact cluster centered at (60, 50)
    - Wire grid extending from plant
    - Road grid on 4-tile intervals (Parent B's winning strategy)
    - Zone placement requiring proper 3x3 clearance
    - Balanced zone mix
    - Services when population warrants
    - Second plant when needed
    """
    tm = obs.tile_map
    em = empty_mask(tm)
    wm = wire_mask(tm)
    rm = road_mask(tm)
    pm = plant_mask(tm)
    resm = res_mask(tm)
    comm = com_mask(tm)
    indm = ind_mask(tm)

    def in_bounds(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    CX, CY = 60, 50  # single cluster center

    # Phase 1: ensure coal power plant
    plant_ys, plant_xs = np.where(pm)
    has_plant = len(plant_xs) > 0
    if not has_plant:
        return (Tool.COALPOWER, CX, CY)

    px = int(plant_xs[0])
    py = int(plant_ys[0])

    pop = obs.city_pop
    res_count = int(np.sum(resm))
    com_count = int(np.sum(comm))
    ind_count = int(np.sum(indm))
    total_zones = max(res_count + com_count + ind_count, 1)
    road_count = int(np.sum(rm))
    wire_count = int(np.sum(wm))

    power_map = pm | wm

    def pick_tool():
        # ~55% R, 25% C, 20% I
        if ind_count * 5 < total_zones:
            return Tool.INDUSTRIAL
        if com_count * 4 < total_zones:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    # ---- Wire growth: expand from existing power tiles ----
    def find_wire_extension():
        power_ys, power_xs = np.where(power_map)
        if len(power_xs) == 0:
            return None

        best = None
        best_score = 1e9

        for i in range(len(power_xs)):
            bx, by = int(power_xs[i]), int(power_ys[i])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                wx, wy = bx + dx, by + dy
                if not in_bounds(wx, wy):
                    continue
                if not bool(em[wy, wx]):
                    continue
                if bool(rm[wy, wx]):
                    continue
                # Cluster radius 28
                if abs(wx - CX) > 28 or abs(wy - CY) > 28:
                    continue
                dist = abs(wx - CX) + abs(wy - CY)
                # Prefer grid lines (every 4 tiles)
                on_hline = (wy - CY) % 4 == 0
                on_vline = (wx - CX) % 4 == 0
                line_bonus = -10 if (on_hline or on_vline) else 0
                score = dist + line_bonus
                if score < best_score:
                    best_score = score
                    best = (Tool.WIRE, wx, wy)
        return best

    # ---- Road growth: grid pattern every 4 tiles ----
    def find_road_extension():
        if road_count == 0:
            # First road adjacent to plant
            for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2),
                           (3, 0), (-3, 0), (0, 3), (0, -3)]:
                rx, ry = px + dx, py + dy
                if in_bounds(rx, ry) and bool(em[ry, rx]) and not bool(pm[ry, rx]):
                    return (Tool.ROAD, rx, ry)
            return None

        road_ys, road_xs = np.where(rm)
        best = None
        best_score = 1e9

        for i in range(len(road_xs)):
            bx, by = int(road_xs[i]), int(road_ys[i])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rx, ry = bx + dx, by + dy
                if not in_bounds(rx, ry):
                    continue
                if not bool(em[ry, rx]):
                    continue
                if bool(pm[ry, rx]) or bool(wm[ry, rx]):
                    continue
                if abs(rx - CX) > 28 or abs(ry - CY) > 28:
                    continue
                # Strong preference for grid lines every 4 tiles from center
                on_hgrid = (ry - CY) % 4 == 0
                on_vgrid = (rx - CX) % 4 == 0
                if not (on_hgrid or on_vgrid):
                    continue
                dist = abs(rx - CX) + abs(ry - CY)
                if dist < best_score:
                    best_score = dist
                    best = (Tool.ROAD, rx, ry)

        if best is not None:
            return best

        # Fallback: any adjacent empty within cluster
        for i in range(len(road_xs)):
            bx, by = int(road_xs[i]), int(road_ys[i])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rx, ry = bx + dx, by + dy
                if not in_bounds(rx, ry):
                    continue
                if not bool(em[ry, rx]):
                    continue
                if bool(pm[ry, rx]) or bool(wm[ry, rx]):
                    continue
                if abs(rx - CX) > 28 or abs(ry - CY) > 28:
                    continue
                dist = abs(rx - CX) + abs(ry - CY)
                if dist < best_score:
                    best_score = dist
                    best = (Tool.ROAD, rx, ry)
        return best

    # ---- Dilation helper ----
    def dilate(mask, radius):
        result = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        ys, xs = np.where(mask)
        for i in range(len(xs)):
            cx2, cy2 = int(xs[i]), int(ys[i])
            r0 = max(0, cy2 - radius)
            r1 = min(WORLD_H, cy2 + radius + 1)
            c0 = max(0, cx2 - radius)
            c1 = min(WORLD_W, cx2 + radius + 1)
            result[r0:r1, c0:c1] = True
        return result

    # ---- Zone placement ----
    def find_zone_placement():
        nbr_road = dilate(rm, 2)
        nbr_power = dilate(power_map, 6)
        zone_mask = resm | comm | indm

        cand = em & nbr_road & nbr_power
        cand[:3, :] = False
        cand[WORLD_H - 3:, :] = False
        cand[:, :3] = False
        cand[:, WORLD_W - 3:] = False
        # Cluster bounds
        cand[:max(0, CY - 26), :] = False
        cand[min(WORLD_H, CY + 27):, :] = False
        cand[:, :max(0, CX - 26)] = False
        cand[:, min(WORLD_W, CX + 27):] = False

        cys, cxs = np.where(cand)
        if len(cxs) == 0:
            return None

        tool = pick_tool()
        zone_dilated = dilate(zone_mask, 3) if total_zones > 1 else np.zeros((WORLD_H, WORLD_W), dtype=bool)

        best = None
        best_score = 1e9

        # Sort by distance from center, check up to 200
        indices = list(range(len(cxs)))
        indices.sort(key=lambda i: abs(int(cxs[i]) - CX) + abs(int(cys[i]) - CY))

        for k in indices[:200]:
            zx, zy = int(cxs[k]), int(cys[k])
            # 3x3 clearance: all tiles must be empty
            ok = True
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    ny2, nx2 = zy + dr, zx + dc
                    if not in_bounds(nx2, ny2):
                        ok = False; break
                    if bool(pm[ny2, nx2]):
                        ok = False; break
                    if bool(rm[ny2, nx2]):
                        ok = False; break
                    if bool(wm[ny2, nx2]):
                        ok = False; break
                    if not bool(em[ny2, nx2]):
                        ok = False; break
                if not ok:
                    break
            if not ok:
                continue

            d = abs(zx - CX) + abs(zy - CY)
            bonus = -6 if bool(zone_dilated[zy, zx]) else 0
            score = d + bonus
            if score < best_score:
                best_score = score
                best = (tool, zx, zy)

        return best

    wire_action = find_wire_extension()
    road_action = find_road_extension()

    # Early bootstrap
    if wire_count < 10 and wire_action is not None:
        return wire_action
    if road_count < 6 and road_action is not None:
        return road_action

    step_mod = obs.step % 12

    # Main phase logic
    if pop < 300:
        if step_mod in (0, 1, 2) and wire_action is not None:
            return wire_action
        if step_mod in (3, 4, 5, 6) and road_action is not None:
            return road_action
        z = find_zone_placement()
        if z is not None:
            return z
        if wire_action is not None:
            return wire_action
        if road_action is not None:
            return road_action

    elif pop < 1000:
        if step_mod in (0, 1) and wire_action is not None:
            return wire_action
        if step_mod in (2, 3, 4) and road_action is not None:
            return road_action
        z = find_zone_placement()
        if z is not None:
            return z
        if wire_action is not None:
            return wire_action
        if road_action is not None:
            return road_action

    else:
        if step_mod == 0 and wire_action is not None:
            return wire_action
        if step_mod in (1, 2) and road_action is not None:
            return road_action
        z = find_zone_placement()
        if z is not None:
            return z
        if wire_action is not None:
            return wire_action
        if road_action is not None:
            return road_action

    # Services
    if pop > 100:
        fire_spots = [
            (CX + 10, CY - 10), (CX - 10, CY + 10),
            (CX + 10, CY + 10), (CX - 10, CY - 10),
            (CX + 14, CY), (CX - 14, CY),
            (CX, CY + 14), (CX, CY - 14),
        ]
        for fx, fy in fire_spots:
            if in_bounds(fx, fy) and bool(em[fy, fx]):
                return (Tool.FIRESTATION, fx, fy)

    if pop > 400:
        police_spots = [
            (CX + 12, CY - 6), (CX - 12, CY + 6),
            (CX + 6, CY + 12), (CX - 6, CY - 12),
        ]
        for fx, fy in police_spots:
            if in_bounds(fx, fy) and bool(em[fy, fx]):
                return (Tool.POLICESTATION, fx, fy)

    # Second plant if large city
    if pop > 1000 and len(plant_xs) < 2:
        for fx, fy in [(CX + 24, CY), (CX - 24, CY),
                       (CX, CY + 24), (CX, CY - 24)]:
            if in_bounds(fx, fy) and bool(em[fy, fx]):
                return (Tool.COALPOWER, fx, fy)

    # Third plant if very large
    if pop > 3000 and len(plant_xs) < 3:
        for fx, fy in [(CX + 24, CY + 24), (CX - 24, CY - 24),
                       (CX + 24, CY - 24), (CX - 24, CY + 24)]:
            if in_bounds(fx, fy) and bool(em[fy, fx]):
                return (Tool.COALPOWER, fx, fy)

    # Final fallback
    z = find_zone_placement()
    if z is not None:
        return z
    if wire_action is not None:
        return wire_action
    if road_action is not None:
        return road_action

    return None
```

## most counterfactual-sensitive (max cf)
- cityPop=1180 (R=21 C=1 I=5) | stored cityPop=2293 | origin=mutate+cl iter=1481
- reactivity: cf_sensitivity=1.00, traj_divergence=0.02, ind_share=0.19  [reactive-map]
- code: 258 lines, 9579 chars | motifs: reads-map, reads-stats, plants=2c, wire, road
```
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
.........................RR.................................
.....................III.RRRR...............................
...................CCIIII*=RRRR.............................
...................CC====.===RR.............................
...................RR........RR.............................
...................RCCRRCCIIRCC.............................
....................CCRRCCIIRCC.............................
............................................................
............................................................
............................................................
............................................................
............................................................
.........................*..................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
```
```python
def act(obs, state):
    """
    Closed-loop reactive policy:
    - Reads tile_map every step
    - Coal plant first, then horizontal wire spine (instead of vertical)
    - Road rows flanking wire spine
    - Reactive zone placement wherever power+road coverage exists
    - Second plant and services as city grows
    """
    tm = obs.tile_map
    em = empty_mask(tm)
    wm = wire_mask(tm)
    rm = road_mask(tm)
    pm = plant_mask(tm)
    resm = res_mask(tm)
    comm = com_mask(tm)
    indm = ind_mask(tm)

    def in_bounds(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    res_count = int(np.sum(resm))
    com_count = int(np.sum(comm))
    ind_count = int(np.sum(indm))
    total_zones = res_count + com_count + ind_count

    def pick_tool():
        if total_zones == 0:
            return Tool.RESIDENTIAL
        t = total_zones
        if ind_count / t < 0.20:
            return Tool.INDUSTRIAL
        if com_count / t < 0.25:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    # --- Phase 1: Place coal power plant if none ---
    if not np.any(pm):
        return (Tool.COALPOWER, 50, 47)

    plant_ys, plant_xs = np.where(pm)
    px = int(np.mean(plant_xs))
    py = int(np.mean(plant_ys))

    pop = obs.city_pop

    # Horizontal wire spine: row just below plant
    wire_row = py + 3
    wire_row = max(3, min(WORLD_H - 4, wire_row))
    road_row_b = wire_row + 2
    road_row_t = wire_row - 2
    road_row_b = max(3, min(WORLD_H - 4, road_row_b))
    road_row_t = max(3, min(WORLD_H - 4, road_row_t))

    # Spine extent grows with population
    if pop < 50:
        spine_half = 8
    elif pop < 150:
        spine_half = 14
    elif pop < 400:
        spine_half = 20
    elif pop < 800:
        spine_half = 28
    else:
        spine_half = 40

    cmin = max(2, px - spine_half)
    cmax = min(WORLD_W - 3, px + spine_half)

    wire_cols = [c for c in range(WORLD_W) if wm[wire_row, c]]

    # --- Phase 2: Bootstrap wire adjacent to plant (horizontally) ---
    if not wire_cols:
        for dc in range(-2, 4):
            nc = px + dc
            if in_bounds(nc, wire_row) and (em[wire_row, nc] or wm[wire_row, nc]) and not pm[wire_row, nc]:
                return (Tool.WIRE, nc, wire_row)
        # try connecting plant to wire row with vertical wire
        for dr in range(1, abs(wire_row - py) + 2):
            nr = py + dr
            if in_bounds(px, nr) and not pm[nr, px]:
                if em[nr, px] or wm[nr, px]:
                    return (Tool.WIRE, px, nr)
        return None

    min_wc = min(wire_cols)
    max_wc = max(wire_cols)

    # --- Phase 3: Extend wire spine rightward ---
    if max_wc < cmax:
        nc = max_wc + 1
        if in_bounds(nc, wire_row) and not pm[wire_row, nc]:
            if em[wire_row, nc] or wm[wire_row, nc]:
                return (Tool.WIRE, nc, wire_row)

    # --- Phase 4: Extend wire spine leftward ---
    if min_wc > cmin:
        nc = min_wc - 1
        if in_bounds(nc, wire_row) and not pm[wire_row, nc]:
            if em[wire_row, nc] or wm[wire_row, nc]:
                return (Tool.WIRE, nc, wire_row)

    # --- Phase 5: Build road rows along current wire extent ---
    for wc in wire_cols:
        if cmin <= wc <= cmax:
            if in_bounds(wc, road_row_b) and em[road_row_b, wc] and not pm[road_row_b, wc]:
                return (Tool.ROAD, wc, road_row_b)
            if in_bounds(wc, road_row_t) and em[road_row_t, wc] and not pm[road_row_t, wc]:
                return (Tool.ROAD, wc, road_row_t)

    # --- Phase 6: Reactive zone placement ---
    nbr_road = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_road[r0:r1, c0:c1] |= rm[sr0:sr1, sc0:sc1]

    nbr_wire = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_wire[r0:r1, c0:c1] |= wm[sr0:sr1, sc0:sc1]
            nbr_wire[r0:r1, c0:c1] |= pm[sr0:sr1, sc0:sc1]

    all_zone_mask = resm | comm | indm

    def zone_clear(zx, zy):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = zy + dr, zx + dc
                if not in_bounds(nc, nr):
                    return False
                if not em[nr, nc] and not all_zone_mask[nr, nc]:
                    return False
        return True

    def zone_occupied(zx, zy):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = zy + dr, zx + dc
                if in_bounds(nc, nr) and all_zone_mask[nr, nc]:
                    return True
        return False

    cand_mask = em & nbr_road & nbr_wire
    cand_mask[0:2, :] = False
    cand_mask[WORLD_H-2:, :] = False
    cand_mask[:, 0:2] = False
    cand_mask[:, WORLD_W-2:] = False

    cys, cxs = np.where(cand_mask)
    if len(cxs) > 0:
        # Prefer positions close to wire row, vary by step for coverage
        dists = np.abs(cys - wire_row) + np.abs(cxs - px) // 3
        order = np.argsort(dists)
        step_offset = int(obs.step * 13) % max(len(cxs), 1)
        for i in range(min(len(cxs), 80)):
            idx = order[(step_offset + i * 7) % len(order)]
            zx, zy = int(cxs[idx]), int(cys[idx])
            if not zone_occupied(zx, zy) and zone_clear(zx, zy):
                return (pick_tool(), zx, zy)

    # --- Phase 7: Lateral wires to extend coverage vertically ---
    for wc in wire_cols:
        if cmin <= wc <= cmax:
            # Extend upward
            for wr in range(wire_row - 1, max(2, wire_row - 8), -1):
                if wm[wr, wc]:
                    break
                if em[wr, wc] and not pm[wr, wc] and not rm[wr, wc]:
                    return (Tool.WIRE, wc, wr)
                elif not em[wr, wc] and not wm[wr, wc]:
                    break
            # Extend downward
            for wr in range(wire_row + 1, min(WORLD_H - 2, wire_row + 8)):
                if wm[wr, wc]:
                    break
                if em[wr, wc] and not pm[wr, wc] and not rm[wr, wc]:
                    return (Tool.WIRE, wc, wr)
                elif not em[wr, wc] and not wm[wr, wc]:
                    break

    # --- Phase 8: Lateral road extensions ---
    road_ys, road_xs = np.where(rm)
    if len(road_xs) > 0:
        step_k = int(obs.step * 11) % len(road_xs)
        for offset in range(min(len(road_xs), 20)):
            idx = (step_k + offset * 7) % len(road_xs)
            rx, ry = int(road_xs[idx]), int(road_ys[idx])
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx2, ny2 = rx + dx, ry + dy
                if in_bounds(nx2, ny2) and em[ny2, nx2] and not pm[ny2, nx2]:
                    if nbr_wire[ny2, nx2]:
                        return (Tool.ROAD, nx2, ny2)

    # --- Phase 9: Second coal plant when population demands it ---
    if pop > 300:
        p2x = px
        p2y = py + 20
        p2x = max(3, min(WORLD_W - 4, p2x))
        p2y = max(3, min(WORLD_H - 4, p2y))
        nearby = False
        for dr in range(-7, 8):
            for dc in range(-7, 8):
                ny2, nx2 = p2y + dr, p2x + dc
                if in_bounds(nx2, ny2) and pm[ny2, nx2]:
                    nearby = True
        if not nearby:
            area_ok = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    ny2, nx2 = p2y + dr, p2x + dc
                    if not in_bounds(nx2, ny2) or (not em[ny2, nx2] and not pm[ny2, nx2]):
                        area_ok = False
            if area_ok:
                return (Tool.COALPOWER, p2x, p2y)
            # Wire down to connect second plant
            for wr in range(wire_row + 3, p2y - 1):
                if in_bounds(px, wr) and em[wr, px]:
                    return (Tool.WIRE, px, wr)

    # --- Phase 10: Services for happiness ---
    if pop > 200:
        fx, fy = px - 8, py - 6
        if in_bounds(fx, fy) and em[fy, fx]:
            return (Tool.FIRESTATION, fx, fy)
        fx2, fy2 = px + 8, py - 6
        if in_bounds(fx2, fy2) and em[fy2, fx2]:
            return (Tool.POLICESTATION, fx2, fy2)

    # --- Phase 11: Wire frontier extension (fallback) ---
    wire_ys2, wire_xs2 = np.where(wm)
    if len(wire_xs2) > 0:
        step_k2 = int(obs.step * 7) % len(wire_xs2)
        wx_s = int(wire_xs2[step_k2])
        wy_s = int(wire_ys2[step_k2])
        for dx2, dy2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx2, ny2 = wx_s + dx2, wy_s + dy2
            if in_bounds(nx2, ny2) and em[ny2, nx2] and not pm[ny2, nx2]:
                return (Tool.WIRE, nx2, ny2)

    # --- Phase 12: Add more road rows for zone coverage ---
    if pop > 100:
        extra_road_rows = [wire_row + 5, wire_row - 5, wire_row + 8, wire_row - 8]
        for rrow in extra_road_rows:
            if not in_bounds(0, rrow):
                continue
            for wc in wire_cols:
                if cmin <= wc <= cmax and in_bounds(wc, rrow) and em[rrow, wc] and not pm[rrow, wc]:
                    return (Tool.ROAD, wc, rrow)

    return None
```

## most trajectory-divergent (max div)
- cityPop=640 (R=0 C=0 I=4) | stored cityPop=640 | origin=mutate iter=5609
- reactivity: cf_sensitivity=0.92, traj_divergence=0.91, ind_share=0.47  [reactive-map]
- code: 235 lines, 9067 chars | motifs: reads-map, reads-stats, plants=1c, wire, road
```
............................................................
............................................................
............................................................
............................................................
...IICC.....................................................
...IICC=...=...=............................................
.....*.=...=..................*.............................
.....==========================.............................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
.....======================.................................
............................................................
.....*........................*.............................
.....=......................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
.....=......................................................
............................................................
.....*......................................................
.....=......................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
```
```python
def act(obs, state):
    """
    Variation: Horizontal cluster layout with 3 rows of clusters.
    Row 1: y=10, Row 2: y=40, Row 3: y=70
    Each row has 3 clusters at x=20, 50, 80
    Wire runs horizontally through each row, roads above/below
    Zone mix: 35% RES, 40% COM, 25% IND (more commercial)
    Two coal plants per row for better power coverage.
    """
    tm = obs.tile_map
    em = empty_mask(tm)
    wm = wire_mask(tm)
    rm = road_mask(tm)
    pm = plant_mask(tm)
    resm = res_mask(tm)
    comm = com_mask(tm)
    indm = ind_mask(tm)

    step = obs.step
    pop = obs.city_pop

    def in_bounds(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    # Horizontal clusters: spine_y is wire row, roads above/below
    # Each cluster: coal plant on left end, wire extends right, roads parallel
    clusters = [
        {'px': 10, 'py': 12, 'spine_y': 12, 'road_y1': 9,  'road_y2': 15, 'max_x': 55},
        {'px': 60, 'py': 12, 'spine_y': 12, 'road_y1': 9,  'road_y2': 15, 'max_x': 110},
        {'px': 10, 'py': 38, 'spine_y': 38, 'road_y1': 35, 'road_y2': 41, 'max_x': 55},
        {'px': 60, 'py': 38, 'spine_y': 38, 'road_y1': 35, 'road_y2': 41, 'max_x': 110},
        {'px': 10, 'py': 64, 'spine_y': 64, 'road_y1': 61, 'road_y2': 67, 'max_x': 55},
        {'px': 60, 'py': 64, 'spine_y': 64, 'road_y1': 61, 'road_y2': 67, 'max_x': 110},
    ]

    # Dynamic expansion: unlock more clusters as population grows
    if pop < 50:
        active_clusters = 1
    elif pop < 150:
        active_clusters = 2
    elif pop < 300:
        active_clusters = 3
    elif pop < 500:
        active_clusters = 4
    elif pop < 800:
        active_clusters = 5
    else:
        active_clusters = 6

    res_count = int(np.sum(resm))
    com_count = int(np.sum(comm))
    ind_count = int(np.sum(indm))
    total_zones = max(res_count + com_count + ind_count, 1)

    def pick_tool():
        r = res_count / total_zones
        c = com_count / total_zones
        i = ind_count / total_zones
        if i < 0.25:
            return Tool.INDUSTRIAL
        if c < 0.40:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    zone_tiles = resm | comm | indm

    def zone_clear(zx, zy):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = zy + dr, zx + dc
                if not in_bounds(nc, nr):
                    return False
                if not em[nr, nc] and not zone_tiles[nr, nc]:
                    return False
        return True

    def handle_cluster(cl):
        px, py = cl['px'], cl['py']
        spine_y = cl['spine_y']
        road_y1 = cl['road_y1']
        road_y2 = cl['road_y2']
        max_x = cl['max_x']

        # 1. Ensure coal plant exists near (px, py)
        plant_found = any(
            in_bounds(px + dx, py + dy) and pm[py + dy, px + dx]
            for dx in range(-2, 3) for dy in range(-2, 3)
        )
        if not plant_found:
            return (Tool.COALPOWER, px, py)

        # Find rightmost wire/plant on spine row
        spine_cols = [c for c in range(WORLD_W) if wm[spine_y, c] or pm[spine_y, c]]
        rightmost_wire = max(spine_cols) if spine_cols else px

        # 2. Extend wire spine rightward
        next_w = rightmost_wire + 1
        if next_w <= max_x and in_bounds(next_w, spine_y) and em[spine_y, next_w]:
            return (Tool.WIRE, next_w, spine_y)

        # 3. Top road - extend rightward where wire is present
        road1_cols = [c for c in range(WORLD_W) if rm[road_y1, c]]
        road1_right = max(road1_cols) if road1_cols else px - 1
        next_r1 = road1_right + 1
        if next_r1 <= rightmost_wire and next_r1 <= max_x and in_bounds(next_r1, road_y1) and em[road_y1, next_r1]:
            return (Tool.ROAD, next_r1, road_y1)

        # 4. Bottom road - extend rightward
        road2_cols = [c for c in range(WORLD_W) if rm[road_y2, c]]
        road2_right = max(road2_cols) if road2_cols else px - 1
        next_r2 = road2_right + 1
        if next_r2 <= rightmost_wire and next_r2 <= max_x and in_bounds(next_r2, road_y2) and em[road_y2, next_r2]:
            return (Tool.ROAD, next_r2, road_y2)

        # 5. Vertical wire branches from spine to roads (every 4 cols)
        for bx in range(px + 3, min(rightmost_wire + 1, max_x + 1), 4):
            if not in_bounds(bx, spine_y):
                continue
            if not (wm[spine_y, bx] or pm[spine_y, bx]):
                continue
            for sign, road_y in [(-1, road_y1), (1, road_y2)]:
                for dy in range(1, abs(spine_y - road_y) + 2):
                    wy = spine_y + sign * dy
                    if not in_bounds(bx, wy):
                        break
                    if rm[wy, bx]:
                        break  # reached road
                    if wm[wy, bx] or pm[wy, bx]:
                        continue
                    if em[wy, bx]:
                        return (Tool.WIRE, bx, wy)
                    break

        # 6. Vertical connector roads every 8 cols between roads
        for vx in range(px + 5, min(rightmost_wire + 1, max_x + 1), 8):
            r1_here = rm[road_y1, vx] if in_bounds(vx, road_y1) else False
            r2_here = rm[road_y2, vx] if in_bounds(vx, road_y2) else False
            if r1_here or r2_here:
                for vy in range(min(road_y1, road_y2) + 1, max(road_y1, road_y2)):
                    if in_bounds(vx, vy) and em[vy, vx]:
                        return (Tool.ROAD, vx, vy)

        return None

    # Cycle through active clusters
    primary = step % max(active_clusters, 1)
    for ci in range(active_clusters):
        result = handle_cluster(clusters[(primary + ci) % active_clusters])
        if result:
            return result

    # --- ZONE PLACEMENT ---
    nbr_road = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_road[r0:r1, c0:c1] |= rm[sr0:sr1, sc0:sc1]

    nbr_wire = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_wire[r0:r1, c0:c1] |= wm[sr0:sr1, sc0:sc1]
            nbr_wire[r0:r1, c0:c1] |= pm[sr0:sr1, sc0:sc1]

    cand_mask = em & nbr_road & nbr_wire
    cand_mask[:3, :] = False
    cand_mask[WORLD_H - 3:, :] = False
    cand_mask[:, :3] = False
    cand_mask[:, WORLD_W - 3:] = False

    cys_arr, cxs_arr = np.where(cand_mask)
    if len(cxs_arr) > 0:
        # Prefer zones near horizontal spine rows
        spine_rows = np.array([12, 38, 64])
        dists = np.min(np.abs(cys_arr[:, None] - spine_rows[None, :]), axis=1)
        # Weight to prefer closer to spine
        score = dists + cxs_arr * 0.1
        order = np.argsort(score)
        offset = (step * 13 + int(pop * 5)) % max(len(order), 1)
        for i in range(min(len(order), 400)):
            k = order[(offset + i) % len(order)]
            zx, zy = int(cxs_arr[k]), int(cys_arr[k])
            if zone_clear(zx, zy):
                return (pick_tool(), zx, zy)

    # --- SERVICES (reactive) ---
    services = state.get('services', set())
    svc_candidates = []
    if pop > 100:
        svc_candidates += [
            ('fs1', Tool.FIRESTATION, 30, 12),
            ('fs2', Tool.FIRESTATION, 80, 12),
        ]
    if pop > 250:
        svc_candidates += [
            ('ps1', Tool.POLICESTATION, 30, 38),
            ('ps2', Tool.POLICESTATION, 80, 38),
        ]
    if pop > 500:
        svc_candidates += [
            ('fs3', Tool.FIRESTATION, 30, 64),
            ('fs4', Tool.FIRESTATION, 80, 64),
            ('ps3', Tool.POLICESTATION, 50, 25),
        ]
    if pop > 800:
        svc_candidates += [
            ('ps4', Tool.POLICESTATION, 50, 50),
            ('ps5', Tool.POLICESTATION, 50, 75),
        ]
    for skey, stool, sx, sy in svc_candidates:
        if skey not in services and in_bounds(sx, sy) and em[sy, sx]:
            services.add(skey)
            state['services'] = services
            return (stool, sx, sy)

    # --- Fallback: extend wire adjacent to existing wire ---
    wire_ys, wire_xs = np.where(wm)
    if len(wire_xs) > 0:
        k = (step * 7) % len(wire_xs)
        for off in range(min(len(wire_xs), 30)):
            idx = (k + off * 3) % len(wire_xs)
            wx, wy = int(wire_xs[idx]), int(wire_ys[idx])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = wx + dx, wy + dy
                if in_bounds(nx, ny) and em[ny, nx] and not pm[ny, nx]:
                    return (Tool.WIRE, nx, ny)

    return None
```

## most industrial
- cityPop=640 (R=0 C=0 I=4) | stored cityPop=640 | origin=mutate+cl iter=3639
- reactivity: cf_sensitivity=0.64, traj_divergence=0.05, ind_share=1.00  [reactive-map]
- code: 287 lines, 10572 chars | motifs: reads-map, reads-stats, plants=2c, wire, road
```
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................II..............................
............................II*.............................
================================............................
.............................==.............................
............................................................
........................................*...................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
```
```python
def act(obs, state):
    """
    Closed-loop policy: reads map each step, extends infrastructure
    from what already exists, places zones where power+road reach.
    Key changes from parent:
    - Single cluster focused around (60, 50) for denser growth
    - Greedy BFS-style wire/road extension always from frontier
    - Zone placement strictly requires zone_clear check and road adjacency
    - More aggressive zone placement relative to infrastructure
    - Service buildings placed reactively based on zone count
    """
    tm = obs.tile_map
    em = empty_mask(tm)
    wm = wire_mask(tm)
    rm = road_mask(tm)
    pm = plant_mask(tm)
    resm = res_mask(tm)
    comm = com_mask(tm)
    indm = ind_mask(tm)

    def in_bounds(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    res_count = int(np.sum(resm)) // 9
    com_count = int(np.sum(comm)) // 9
    ind_count = int(np.sum(indm)) // 9
    total_zones = max(res_count + com_count + ind_count, 1)
    pop = obs.city_pop
    wire_count = int(np.sum(wm))
    road_count = int(np.sum(rm))

    # Central cluster
    CX, CY = 60, 50
    ROAD_ROW1 = 44
    ROAD_ROW2 = 56

    def pick_tool():
        if ind_count * 4 < total_zones:
            return Tool.INDUSTRIAL
        if com_count * 3 < total_zones:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    def zone_clear(zx, zy):
        zone_tiles = resm | comm | indm
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = zy + dr, zx + dc
                if not in_bounds(nc, nr):
                    return False
                if not em[nr, nc] and not zone_tiles[nr, nc]:
                    return False
        return True

    def expand_mask(base, radius):
        result = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) > radius + 1:
                    continue
                sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
                sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
                r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
                c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
                result[r0:r1, c0:c1] |= base[sr0:sr1, sc0:sc1]
        return result

    plant_ys, plant_xs = np.where(pm)
    n_plants = len(plant_xs)
    power_source = pm | wm

    # --- PHASE 0: Place coal plant ---
    if n_plants == 0:
        return (Tool.COALPOWER, CX, CY - 8)

    # --- PHASE 1: Second plant when pop warrants ---
    if n_plants < 18 and pop > 150:
        px2, py2 = CX + 20, CY
        if in_bounds(px2, py2) and bool(em[py2, px2]):
            return (Tool.COALPOWER, px2, py2)

    # --- WIRE EXTENSION: grow from power source frontier ---
    def best_wire():
        py_arr, px_arr = np.where(power_source)
        if len(px_arr) == 0:
            return None
        best, best_score = None, -1
        idxs = list(range(len(px_arr)))
        if len(idxs) > 400:
            step = len(idxs) // 400
            idxs = idxs[::step]
        for idx in idxs:
            ppx, ppy = int(px_arr[idx]), int(py_arr[idx])
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                wx, wy = ppx + ddx, ppy + ddy
                if not in_bounds(wx, wy): continue
                if bool(pm[wy, wx]) or bool(wm[wy, wx]): continue
                if not bool(em[wy, wx]): continue
                score = 0
                # Prefer rows near road rows (to power zones adjacent to roads)
                if abs(wy - ROAD_ROW1) <= 3: score += 25
                if abs(wy - ROAD_ROW2) <= 3: score += 25
                # Prefer horizontal movement toward center x range
                if 30 <= wx <= 90: score += 15
                # Closer to cluster center
                d = abs(wx - CX) + abs(wy - CY)
                score += max(0, 40 - d) * 0.4
                # Neighbor of existing road = good
                for ddx2, ddy2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx2, ny2 = wx + ddx2, wy + ddy2
                    if in_bounds(nx2, ny2) and bool(rm[ny2, nx2]):
                        score += 8
                if score > best_score:
                    best_score = score
                    best = (Tool.WIRE, wx, wy)
        return best

    # --- ROAD EXTENSION: grow along two horizontal corridors ---
    def best_road():
        ry_arr, rx_arr = np.where(rm)
        nbr_power = expand_mask(power_source, 4)
        best, best_score = None, -1

        # Bootstrap: no roads yet, start near cluster
        if len(rx_arr) == 0:
            for dc in range(-10, 11):
                for row in [ROAD_ROW1, ROAD_ROW2]:
                    nx, ny = CX + dc, row
                    if in_bounds(nx, ny) and bool(em[ny, nx]) and not bool(pm[ny, nx]):
                        if bool(nbr_power[ny, nx]):
                            return (Tool.ROAD, nx, ny)
            return None

        idxs = list(range(len(rx_arr)))
        if len(idxs) > 400:
            step = len(idxs) // 400
            idxs = idxs[::step]

        for idx in idxs:
            rx2, ry2 = int(rx_arr[idx]), int(ry_arr[idx])
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx2, ny2 = rx2 + ddx, ry2 + ddy
                if not in_bounds(nx2, ny2): continue
                if bool(rm[ny2, nx2]) or bool(pm[ny2, nx2]): continue
                if not bool(em[ny2, nx2]): continue
                score = 0
                # Strongly prefer staying in road rows
                if ny2 == ROAD_ROW1: score += 30
                if ny2 == ROAD_ROW2: score += 30
                if ry2 == ROAD_ROW1 and ny2 == ROAD_ROW1: score += 15
                if ry2 == ROAD_ROW2 and ny2 == ROAD_ROW2: score += 15
                # Prefer horizontal range around cluster
                if 20 <= nx2 <= 100: score += 10
                d = abs(nx2 - CX) + abs(ny2 - CY)
                score += max(0, 45 - d) * 0.3
                # Need power nearby
                if not bool(nbr_power[ny2, nx2]):
                    score -= 40
                # Near zones
                for ddx2, ddy2 in [(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)]:
                    nx3, ny3 = nx2 + ddx2, ny2 + ddy2
                    if in_bounds(nx3, ny3):
                        if bool(resm[ny3, nx3]) or bool(comm[ny3, nx3]) or bool(indm[ny3, nx3]):
                            score += 5
                if score > best_score:
                    best_score = score
                    best = (Tool.ROAD, nx2, ny2)
        return best

    # --- ZONE PLACEMENT ---
    def place_zone():
        nbr_road = expand_mask(rm, 2)
        nbr_power = expand_mask(power_source, 3)

        cand = em & nbr_road & nbr_power
        # Keep border buffer
        cand[:3, :] = False; cand[WORLD_H-3:, :] = False
        cand[:, :3] = False; cand[:, WORLD_W-3:] = False

        cys, cxs = np.where(cand)
        if len(cxs) == 0:
            return None

        scores = np.zeros(len(cxs), dtype=float)
        d = np.abs(cxs - CX) + np.abs(cys - CY)
        scores -= d * 0.15

        # Prefer near road rows (zones can get road access)
        scores += (np.abs(cys - ROAD_ROW1) <= 4).astype(float) * 8
        scores += (np.abs(cys - ROAD_ROW2) <= 4).astype(float) * 8

        # Prefer directly adjacent to road
        road_near = expand_mask(rm, 1)
        for i in range(len(cxs)):
            if bool(road_near[cys[i], cxs[i]]):
                scores[i] += 12

        tool = pick_tool()
        order = np.argsort(-scores)
        for i in range(min(len(cxs), 200)):
            k = order[i]
            zx, zy = int(cxs[k]), int(cys[k])
            if zone_clear(zx, zy):
                return (tool, zx, zy)
        return None

    # --- SERVICE BUILDINGS (reactive to zone count) ---
    def place_services():
        spots_fire = [
            (CX - 12, CY - 10), (CX + 12, CY - 10),
            (CX - 12, CY + 10), (CX + 12, CY + 10),
        ]
        spots_police = [
            (CX, CY - 12), (CX, CY + 12),
            (CX - 15, CY), (CX + 15, CY),
        ]
        if total_zones > 5:
            for fx, fy in spots_fire[:2]:
                fx = max(2, min(WORLD_W - 3, fx))
                fy = max(2, min(WORLD_H - 3, fy))
                if bool(em[fy, fx]):
                    return (Tool.FIRESTATION, fx, fy)
        if total_zones > 15:
            for px2, py2 in spots_police[:2]:
                px2 = max(2, min(WORLD_W - 3, px2))
                py2 = max(2, min(WORLD_H - 3, py2))
                if bool(em[py2, px2]):
                    return (Tool.POLICESTATION, px2, py2)
        if total_zones > 25:
            for fx, fy in spots_fire[2:]:
                fx = max(2, min(WORLD_W - 3, fx))
                fy = max(2, min(WORLD_H - 3, fy))
                if bool(em[fy, fx]):
                    return (Tool.FIRESTATION, fx, fy)
        if total_zones > 35:
            for px2, py2 in spots_police[2:]:
                px2 = max(2, min(WORLD_W - 3, px2))
                py2 = max(2, min(WORLD_H - 3, py2))
                if bool(em[py2, px2]):
                    return (Tool.POLICESTATION, px2, py2)
        return None

    # --- MAIN LOGIC ---
    wire_action = best_wire()
    road_action = best_road()

    # Services first (reactive)
    svc = place_services()
    if svc:
        return svc

    # Bootstrap: must have wire and road minimums
    if wire_count < 15:
        if wire_action: return wire_action
    if road_count < 15:
        if road_action: return road_action

    step_mod = obs.step % 6

    if pop < 30:
        # Heavy infrastructure
        if step_mod < 2 and wire_action: return wire_action
        if step_mod < 5 and road_action: return road_action
        za = place_zone()
        if za: return za
    elif pop < 150:
        # Balance
        if step_mod == 0 and wire_action: return wire_action
        if step_mod in (1, 2) and road_action: return road_action
        za = place_zone()
        if za: return za
        if road_action: return road_action
    else:
        # Zone dominant
        za = place_zone()
        if za: return za
        if step_mod == 0 and wire_action: return wire_action
        if step_mod == 1 and road_action: return road_action
        za = place_zone()
        if za: return za

    # Fallbacks
    za = place_zone()
    if za: return za
    if wire_action: return wire_action
    if road_action: return road_action
    return None
```

