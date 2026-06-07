# Deep analysis — elm_clind_fine
archive: 256 cells, qd=237240, best cityPop=4280

## champion (max cityPop)
- cityPop=2720 (R=96 C=4 I=0) | stored cityPop=4280 | origin=crossover+cl iter=7131
- reactivity: cf_sensitivity=0.88, traj_divergence=0.38, ind_share=0.15  [reactive-map]
- code: 255 lines, 9592 chars | motifs: reads-map, reads-stats, plants=2c, wire, road
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
....CC==II=.................................................
....CCR.II..................................................
....RRR====.................................................
....RRRRCC..................................................
....RRRRCC..................................................
....*.......................................................
......RCC...................................................
....CCRCC==.................................................
....CCRRR...................................................
....IIRRR...................................................
....II=====.................................................
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
    tm = obs.tile_map
    emp = empty_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    wires = wire_mask(tm)
    roads = road_mask(tm)
    plants = plant_mask(tm)

    step = int(obs.step)
    power_grid = plants | wires
    zone_any = res | com | ind

    def inb(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    def near_mask(mask, x, y, dist=1):
        y0 = max(0, y - dist); y1 = min(WORLD_H, y + dist + 1)
        x0 = max(0, x - dist); x1 = min(WORLD_W, x + dist + 1)
        return bool(np.any(mask[y0:y1, x0:x1]))

    def adj4(y, x):
        return [(y+dy, x+dx) for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                if inb(x+dx, y+dy)]

    def adj_has(mask, y, x):
        return any(mask[ny, nx] for ny, nx in adj4(y, x))

    def expand_mask(mask):
        out = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        out[1:, :]  |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:]  |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    # Layout: coal plant at left-center, wire spine horizontal,
    # vertical wire branches, horizontal road rows, zone fill
    PLANT_X = 8
    PLANT_Y = 50
    SPINE_REACH = 90
    CROSS_SPACING = 6
    ROAD_OFFSETS = [-5, 5, -10, 10, -15, 15, -20, 20, -25, 25]

    plant_ys, plant_xs = np.where(plants)
    n_plants = int(len(plant_xs))

    # === PHASE 0: Place first coal power plant ===
    if n_plants == 0:
        return (Tool.COALPOWER, PLANT_X, PLANT_Y)

    px = int(plant_xs[0])
    py = int(plant_ys[0])
    spine_y = py
    spine_start = px + 3
    spine_end = min(px + SPINE_REACH, WORLD_W - 4)

    road_rows = [spine_y + off for off in ROAD_OFFSETS
                 if 3 <= spine_y + off < WORLD_H - 3]

    # === PHASE 1: Extend wire spine rightward (reactive) ===
    # Find rightmost powered cell on spine row
    spine_powered = power_grid[spine_y, :]
    powered_cols = np.where(spine_powered)[0]
    rightmost_spine = int(np.max(powered_cols)) if len(powered_cols) > 0 else px

    if rightmost_spine < spine_end:
        nx = rightmost_spine + 1
        if inb(nx, spine_y) and emp[spine_y, nx]:
            return (Tool.WIRE, nx, spine_y)

    # === PHASE 2: Reactive extra power plants based on city growth ===
    # Add more plants as population or step grows
    pop = int(obs.city_pop)
    pz = int(obs.powered_zones)

    extra_plants = [
        (1, 35, pop > 50 or step > 20, spine_y),
        (2, 60, pop > 200 or step > 100, spine_y),
        (3, 80, pop > 500 or step > 220, spine_y),
        (4, 45, pop > 900 or step > 380, spine_y - 25),
        (5, 65, pop > 1500 or step > 500, spine_y - 25),
    ]
    for needed_count, offset_x, condition, ty in extra_plants:
        if n_plants == needed_count and condition:
            tx = px + offset_x
            if inb(tx, ty) and emp[ty, tx]:
                if near_mask(power_grid, tx, ty, 15):
                    return (Tool.COALPOWER, tx, ty)

    # === PHASE 3: Vertical wire branches from spine to road rows (reactive) ===
    # For each powered column on spine, extend wire toward each road row
    for bx in range(spine_start, min(rightmost_spine + 1, spine_end + 1), CROSS_SPACING):
        if not inb(bx, spine_y):
            continue
        if not power_grid[spine_y, bx]:
            continue
        for ry in road_rows:
            if ry > spine_y:
                # Extend downward
                for vy in range(spine_y + 1, min(ry + 2, WORLD_H - 1)):
                    if not inb(bx, vy):
                        break
                    if emp[vy, bx] and power_grid[vy - 1, bx]:
                        return (Tool.WIRE, bx, vy)
                    elif not power_grid[vy, bx]:
                        break
            elif ry < spine_y:
                # Extend upward
                for vy in range(spine_y - 1, max(ry - 1, 0), -1):
                    if not inb(bx, vy):
                        break
                    if emp[vy, bx] and power_grid[vy + 1, bx]:
                        return (Tool.WIRE, bx, vy)
                    elif not power_grid[vy, bx]:
                        break

    # === PHASE 4: Build horizontal road rows (reactive, extend from current end) ===
    for ry in road_rows:
        if ry < 2 or ry >= WORLD_H - 2:
            continue
        road_row_cols = np.where(roads[ry, :])[0]
        if len(road_row_cols) > 0:
            start_rx = int(np.max(road_row_cols)) + 1
        else:
            start_rx = spine_start

        for rx in range(start_rx, spine_end + 5):
            if not inb(rx, ry):
                break
            if roads[ry, rx]:
                continue
            if not emp[ry, rx]:
                break
            if near_mask(power_grid, rx, ry, 12):
                return (Tool.ROAD, rx, ry)
            else:
                break

    # === PHASE 5: Vertical connector roads between road rows ===
    valid_rows = sorted([ry for ry in road_rows if 2 <= ry < WORLD_H - 2])
    if len(valid_rows) >= 2:
        r_min = min(valid_rows)
        r_max = max(valid_rows)
        for cx in range(spine_start + CROSS_SPACING, spine_end + 3, CROSS_SPACING):
            if not inb(cx, spine_y):
                continue
            if not near_mask(power_grid, cx, spine_y, 5):
                continue
            for cy in range(r_min, r_max + 1):
                if not inb(cx, cy):
                    continue
                if cy == spine_y:
                    continue
                if emp[cy, cx] and near_mask(roads, cx, cy, 1):
                    if near_mask(power_grid, cx, cy, 8):
                        return (Tool.ROAD, cx, cy)

    # === PHASE 6: Zone placement with RCI balance (reactive scan) ===
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))
    total_zones = total_res + total_com + total_ind

    # Target ~4:2:1 R:C:I
    if total_res == 0:
        desired = Tool.RESIDENTIAL
    elif total_com == 0:
        desired = Tool.COMMERCIAL
    elif total_ind == 0:
        desired = Tool.INDUSTRIAL
    else:
        r_frac = total_res / max(1, total_zones)
        c_frac = total_com / max(1, total_zones)
        i_frac = total_ind / max(1, total_zones)
        if r_frac < 0.57:
            desired = Tool.RESIDENTIAL
        elif c_frac < 0.28:
            desired = Tool.COMMERCIAL
        elif i_frac < 0.14:
            desired = Tool.INDUSTRIAL
        else:
            c = step % 7
            desired = Tool.RESIDENTIAL if c < 4 else (Tool.COMMERCIAL if c < 6 else Tool.INDUSTRIAL)

    center_x = (spine_start + spine_end) // 2

    # Scan near roads for valid zone positions
    road_ys, road_xs = np.where(roads)
    candidates = []

    if len(road_xs) > 0:
        checked = set()
        for i in range(len(road_xs)):
            rx2, ry2 = int(road_xs[i]), int(road_ys[i])
            for dz in range(-5, 6):
                for dzy in range(-5, 6):
                    zx, zy = rx2 + dz, ry2 + dzy
                    if (zx, zy) in checked:
                        continue
                    checked.add((zx, zy))
                    if zx < 2 or zx >= WORLD_W - 2 or zy < 2 or zy >= WORLD_H - 2:
                        continue
                    if zy == spine_y:
                        continue
                    if not emp[zy, zx]:
                        continue
                    if not near_mask(roads, zx, zy, 2):
                        continue
                    if not near_mask(power_grid, zx, zy, 9):
                        continue
                    y0, y1 = max(0, zy - 1), min(WORLD_H, zy + 2)
                    x0, x1 = max(0, zx - 1), min(WORLD_W, zx + 2)
                    region = emp[y0:y1, x0:x1]
                    if region.shape[0] < 3 or region.shape[1] < 3:
                        continue
                    if int(np.sum(region)) < 7:
                        continue
                    road_adj = 1 if adj_has(roads, zy, zx) else 0
                    near_zone = 1 if near_mask(zone_any, zx, zy, 3) else 0
                    powered_close = 1 if near_mask(power_grid, zx, zy, 4) else 0
                    dist_c = abs(zx - center_x) + abs(zy - spine_y) * 0.6
                    score = dist_c - road_adj * 20 - powered_close * 10 + near_zone * 4
                    candidates.append((score, zx, zy))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        top_n = max(1, min(30, len(candidates)))
        # Vary selection to avoid repeating same spot every step
        idx = int((pop + pz + step) % top_n)
        _, zx, zy = candidates[idx]
        return (desired, zx, zy)

    # === PHASE 7: Wire frontier expansion (reactive fallback) ===
    pg_frontier = expand_mask(power_grid) & emp
    wf_ys, wf_xs = np.where(pg_frontier)
    if len(wf_xs) > 0:
        target_y = road_rows[0] if road_rows else spine_y + 5
        dists = np.abs(wf_xs - center_x).astype(float) + np.abs(wf_ys - target_y) * 0.5
        idx = int(np.argmin(dists))
        return (Tool.WIRE, int(wf_xs[idx]), int(wf_ys[idx]))

    # === PHASE 8: Road frontier expansion (reactive fallback) ===
    rd_frontier = expand_mask(roads) & emp
    rf_ys, rf_xs = np.where(rd_frontier)
    if len(rf_xs) > 0:
        dists = np.abs(rf_xs - center_x) + np.abs(rf_ys - spine_y)
        order = np.argsort(dists)
        for idx2 in order[:20]:
            rx2, ry2 = int(rf_xs[idx2]), int(rf_ys[idx2])
            if near_mask(power_grid, rx2, ry2, 12):
                return (Tool.ROAD, rx2, ry2)

    return None
```

## most counterfactual-sensitive (max cf)
- cityPop=1040 (R=60 C=1 I=1) | stored cityPop=1533 | origin=mutate+cl iter=5671
- reactivity: cf_sensitivity=1.00, traj_divergence=0.01, ind_share=0.21  [reactive-map]
- code: 263 lines, 10584 chars | motifs: reads-map, reads-stats, plants=4c, wire, road
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
=RRRIIIRR===................................................
.RRRIIIRR...................................................
....RRRR===.................................................
....RRRR=...................................................
....RRRR===.................................................
....RRRR=...................................................
....*CCII...................................................
....RCCII...................................................
....RRCC===.................................................
....CCCC=...................................................
....CCRR===.................................................
....RRRR=...................................................
.....======.................................................
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
    tm = obs.tile_map
    emp = empty_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    wires = wire_mask(tm)
    roads = road_mask(tm)
    plants = plant_mask(tm)

    step = int(obs.step)
    power_grid = plants | wires

    def in_bounds(y, x):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    def near_mask(mask, x, y, dist=1):
        y0 = max(0, y - dist); y1 = min(WORLD_H, y + dist + 1)
        x0 = max(0, x - dist); x1 = min(WORLD_W, x + dist + 1)
        return bool(np.any(mask[y0:y1, x0:x1]))

    def adj4(y, x):
        return [(y+dy, x+dx) for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                if in_bounds(y+dy, x+dx)]

    def adj_has(mask, y, x):
        return any(mask[ny, nx] for ny, nx in adj4(y, x))

    # Layout constants - compact city centered around y=50
    PLANT_X = 8
    PLANT_Y = 50
    SPINE_Y = 50
    SPINE_START_OFFSET = 3
    SPINE_REACH = 80
    CROSS_SPACING = 6
    # Multiple road rows for more zone coverage
    ROAD_OFFSETS = [-4, 4, -8, 8, -12, 12]

    plant_ys, plant_xs = np.where(plants)
    n_plants = int(len(plant_xs))

    # Phase 0: Place first coal power plant
    if n_plants == 0:
        return (Tool.COALPOWER, PLANT_X, PLANT_Y)

    # Anchor to actual plant position
    px = int(plant_xs[0])
    py = int(plant_ys[0])
    spine_y = py
    spine_start = px + SPINE_START_OFFSET
    spine_end = min(px + SPINE_REACH, WORLD_W - 4)
    road_rows = [spine_y + off for off in ROAD_OFFSETS if 2 <= spine_y + off < WORLD_H - 2]

    # Find current wire spine extent (closed-loop)
    spine_row_mask = wires[spine_y, :] | plants[spine_y, :]
    if np.any(spine_row_mask):
        rightmost_spine = int(np.max(np.where(spine_row_mask)[0]))
    else:
        rightmost_spine = px

    # --- Phase 1: Extend wire spine rightward ---
    if rightmost_spine < spine_end:
        nx = rightmost_spine + 1
        if in_bounds(spine_y, nx) and emp[spine_y, nx] and near_mask(power_grid, nx, spine_y, 1):
            return (Tool.WIRE, nx, spine_y)

    # --- Phase 2: Reactive extra power plants based on actual pop ---
    if n_plants == 1 and obs.city_pop > 30 and step > 15:
        mid_x = px + 30
        if in_bounds(py, mid_x) and emp[py, mid_x] and near_mask(power_grid, mid_x, py, 8):
            return (Tool.COALPOWER, mid_x, py)
    if n_plants == 2 and obs.city_pop > 120 and step > 50:
        mid_x = px + 55
        if in_bounds(py, mid_x) and emp[py, mid_x] and near_mask(power_grid, mid_x, py, 8):
            return (Tool.COALPOWER, mid_x, py)
    if n_plants == 3 and obs.city_pop > 300 and step > 120:
        mid_x = px + 75
        if in_bounds(py, mid_x) and emp[py, mid_x] and near_mask(power_grid, mid_x, py, 8):
            return (Tool.COALPOWER, mid_x, py)

    # --- Phase 3: Drop vertical wires from spine to road rows (closed-loop) ---
    for bx in range(spine_start, min(rightmost_spine + 1, spine_end + 1), CROSS_SPACING):
        if not in_bounds(spine_y, bx):
            continue
        if not (wires[spine_y, bx] or plants[spine_y, bx]):
            continue
        for ry in road_rows:
            if ry > spine_y:
                for vy in range(spine_y + 1, min(ry + 1, WORLD_H - 1)):
                    if emp[vy, bx] and near_mask(power_grid, bx, vy, 1):
                        return (Tool.WIRE, bx, vy)
                    elif not emp[vy, bx] and not power_grid[vy, bx]:
                        break
            elif ry < spine_y:
                for vy in range(spine_y - 1, max(ry - 1, 0), -1):
                    if emp[vy, bx] and near_mask(power_grid, bx, vy, 1):
                        return (Tool.WIRE, bx, vy)
                    elif not emp[vy, bx] and not power_grid[vy, bx]:
                        break

    # --- Phase 4: Build horizontal road rows (closed-loop: observe actual road extent) ---
    for ry in road_rows:
        if ry < 2 or ry >= WORLD_H - 2:
            continue
        road_row_mask = roads[ry, :]
        if np.any(road_row_mask):
            rightmost_road = int(np.max(np.where(road_row_mask)[0]))
        else:
            rightmost_road = spine_start - 1
        nrx = rightmost_road + 1
        if nrx <= spine_end + 2 and in_bounds(ry, nrx) and emp[ry, nrx]:
            if near_mask(power_grid, nrx, ry, 12):
                return (Tool.ROAD, nrx, ry)

    # --- Phase 5: Vertical cross-streets connecting road rows ---
    for cx in range(spine_start + CROSS_SPACING, spine_end + 3, CROSS_SPACING):
        if not in_bounds(0, cx):
            continue
        valid = [ry for ry in road_rows if 2 <= ry < WORLD_H - 2]
        if len(valid) < 2:
            continue
        r_min = min(valid)
        r_max = max(valid)
        for ry2 in range(r_min, r_max + 1):
            if not in_bounds(ry2, cx):
                continue
            if emp[ry2, cx] and near_mask(roads, cx, ry2, 1):
                return (Tool.ROAD, cx, ry2)

    # --- Phase 6: Zone placement with closed-loop RCI balance ---
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))

    # Adaptive RCI ratio: aim for 3:1:1 res:com:ind
    if total_res == 0:
        desired = Tool.RESIDENTIAL
    elif total_com * 3 < total_res:
        desired = Tool.COMMERCIAL
    elif total_ind * 3 < total_res:
        desired = Tool.INDUSTRIAL
    else:
        desired = Tool.RESIDENTIAL

    zone_any = res | com | ind
    candidates = []

    # Scan rows near road rows - use actual road map to find good spots
    # Find all road tiles and scan nearby empty cells
    road_ys, road_xs = np.where(roads)
    
    if len(road_xs) > 0:
        # For each road tile, check adjacent cells as zone centers
        checked = set()
        for i in range(len(road_xs)):
            rx, ry = int(road_xs[i]), int(road_ys[i])
            for dz in range(-4, 5):
                for dzy in range(-4, 5):
                    zx, zy = rx + dz, ry + dzy
                    if (zx, zy) in checked:
                        continue
                    checked.add((zx, zy))
                    if zx < 2 or zx >= WORLD_W - 2 or zy < 2 or zy >= WORLD_H - 2:
                        continue
                    if zy == spine_y:
                        continue
                    if not emp[zy, zx]:
                        continue
                    if not near_mask(roads, zx, zy, 2):
                        continue
                    if not near_mask(power_grid, zx, zy, 10):
                        continue
                    # Check 3x3 footprint is mostly empty
                    y0, y1 = max(0, zy - 1), min(WORLD_H, zy + 2)
                    x0, x1 = max(0, zx - 1), min(WORLD_W, zx + 2)
                    region = emp[y0:y1, x0:x1]
                    if region.shape[0] < 3 or region.shape[1] < 3:
                        continue
                    if int(np.sum(region)) < 7:
                        continue
                    # Score: prefer road adjacency, power proximity, spread out
                    road_adj = 1 if adj_has(roads, zy, zx) else 0
                    near_zone = 1 if near_mask(zone_any, zx, zy, 3) else 0
                    powered_close = 1 if near_mask(power_grid, zx, zy, 4) else 0
                    dist_spine = abs(zy - spine_y)
                    # Prefer cells adjacent to road, close to power, not too crowded
                    score = -road_adj * 10 - powered_close * 5 + near_zone * 6 + dist_spine // 3
                    candidates.append((score, zx, zy))
    else:
        # Fallback: scan valid rows
        valid_rows = []
        for ry in road_rows:
            for offset in range(-4, 5):
                yr = ry + offset
                if 2 <= yr < WORLD_H - 2 and yr != spine_y:
                    valid_rows.append(yr)
        valid_rows = sorted(set(valid_rows))

        for zy in valid_rows:
            for zx in range(spine_start - 2, spine_end + 4):
                if zx < 2 or zx >= WORLD_W - 2:
                    continue
                if not emp[zy, zx]:
                    continue
                if not near_mask(roads, zx, zy, 2):
                    continue
                if not near_mask(power_grid, zx, zy, 10):
                    continue
                y0, y1 = max(0, zy - 1), min(WORLD_H, zy + 2)
                x0, x1 = max(0, zx - 1), min(WORLD_W, zx + 2)
                region = emp[y0:y1, x0:x1]
                if region.shape[0] < 3 or region.shape[1] < 3:
                    continue
                if int(np.sum(region)) < 7:
                    continue
                road_adj = 1 if adj_has(roads, zy, zx) else 0
                near_zone = 1 if near_mask(zone_any, zx, zy, 4) else 0
                powered_close = 1 if near_mask(power_grid, zx, zy, 3) else 0
                dist_spine = abs(zy - spine_y)
                score = -road_adj * 10 - powered_close * 5 + near_zone * 8 + dist_spine // 2
                candidates.append((score, zx, zy))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        top_n = max(1, min(30, len(candidates)))
        idx = step % top_n
        _, zx, zy = candidates[idx]
        return (desired, zx, zy)

    # --- Fallback: extend wire frontier toward road area ---
    pg_adj = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    pg_adj[1:, :]  |= power_grid[:-1, :]
    pg_adj[:-1, :] |= power_grid[1:, :]
    pg_adj[:, 1:]  |= power_grid[:, :-1]
    pg_adj[:, :-1] |= power_grid[:, 1:]
    pg_frontier = pg_adj & emp

    wf_ys, wf_xs = np.where(pg_frontier)
    if len(wf_xs) > 0:
        target_x = (px + spine_end) // 2
        target_y = road_rows[0] if road_rows else spine_y + 4
        dists = np.abs(wf_xs - target_x) + np.abs(wf_ys - target_y)
        idx = int(np.argmin(dists))
        return (Tool.WIRE, int(wf_xs[idx]), int(wf_ys[idx]))

    # --- Last resort: extend road frontier ---
    rd_adj = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    rd_adj[1:, :]  |= roads[:-1, :]
    rd_adj[:-1, :] |= roads[1:, :]
    rd_adj[:, 1:]  |= roads[:, :-1]
    rd_adj[:, :-1] |= roads[:, 1:]
    rd_frontier = rd_adj & emp
    rf_ys, rf_xs = np.where(rd_frontier)
    if len(rf_xs) > 0:
        valid_rrows = [ry for ry in road_rows if 2 <= ry < WORLD_H - 2]
        if valid_rrows:
            dists = np.min(np.stack([np.abs(rf_ys - ry) for ry in valid_rrows], axis=0), axis=0)
            idx = int(np.argmin(dists))
            rx2, ry2 = int(rf_xs[idx]), int(rf_ys[idx])
            if near_mask(power_grid, rx2, ry2, 15):
                return (Tool.ROAD, rx2, ry2)

    return None
```

## most trajectory-divergent (max div)
- cityPop=640 (R=0 C=0 I=5) | stored cityPop=693 | origin=crossover iter=6045
- reactivity: cf_sensitivity=0.95, traj_divergence=0.84, ind_share=0.19  [reactive-map]
- code: 231 lines, 8407 chars | motifs: reads-map, reads-stats, plants=4c, wire, road
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
............................................................
......................CCRR........RR.RII..........RCCRRRR...
......................CCRR........RR=RII=........=RCCRRRRII.
..................................RR=.=RR...........=.RR.II.
......................*.RR........RR=*.RR.........RR*.RRRR*.
......................I=RR........=======........=RR====RR..
......................IIRR.........CCRRII........II.RRCCR...
........................RR.........CCRRII........II.RRCCR...
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
    tm = obs.tile_map
    emp = empty_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    wires = wire_mask(tm)
    roads = road_mask(tm)
    plants = plant_mask(tm)

    step = int(obs.step)
    power_grid = plants | wires

    def in_bounds(y, x):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    def near_mask(mask, x, y, dist=1):
        y0 = max(0, y - dist); y1 = min(WORLD_H, y + dist + 1)
        x0 = max(0, x - dist); x1 = min(WORLD_W, x + dist + 1)
        return np.any(mask[y0:y1, x0:x1])

    def count_mask(mask, x, y, dist=1):
        y0 = max(0, y - dist); y1 = min(WORLD_H, y + dist + 1)
        x0 = max(0, x - dist); x1 = min(WORLD_W, x + dist + 1)
        return int(np.sum(mask[y0:y1, x0:x1]))

    # ---- Phase 0: Place first coal power plant ----
    plant_ys, plant_xs = np.where(plants)
    n_plants = len(plant_xs)

    if n_plants == 0:
        # Place near center for good coverage
        return (Tool.COALPOWER, 45, 50)

    px0, py0 = int(plant_xs[0]), int(plant_ys[0])
    spine_y = py0
    spine_end_x = min(px0 + 65, WORLD_W - 4)

    # ---- Phase 1: Build horizontal wire spine from plant rightward ----
    spine_row_pg = wires[spine_y, :] | plants[spine_y, :]
    if np.any(spine_row_pg):
        rightmost_spine = int(np.max(np.where(spine_row_pg)[0]))
    else:
        rightmost_spine = px0

    if rightmost_spine < spine_end_x:
        nx = rightmost_spine + 1
        if in_bounds(spine_y, nx) and emp[spine_y, nx]:
            return (Tool.WIRE, nx, spine_y)
        elif in_bounds(spine_y, nx) and roads[spine_y, nx]:
            for nx2 in range(nx + 1, spine_end_x + 1):
                if in_bounds(spine_y, nx2) and emp[spine_y, nx2]:
                    return (Tool.WIRE, nx2, spine_y)

    # ---- Phase 2: Add more power plants as city grows ----
    if n_plants == 1 and (obs.city_pop > 50 or step > 60):
        far = plant_xs[plant_xs > px0 + 25]
        if len(far) == 0:
            spx = min(px0 + 30, WORLD_W - 4)
            if in_bounds(spine_y, spx) and not plants[spine_y, spx]:
                return (Tool.COALPOWER, spx, spine_y)

    if n_plants == 2 and (obs.city_pop > 200 or step > 150):
        far = plant_xs[plant_xs > px0 + 55]
        if len(far) == 0:
            spx = min(px0 + 60, WORLD_W - 4)
            if in_bounds(spine_y, spx) and not plants[spine_y, spx]:
                return (Tool.COALPOWER, spx, spine_y)

    if n_plants == 3 and (obs.city_pop > 500 or step > 250):
        far = plant_xs[plant_xs > px0 + 85]
        if len(far) == 0:
            spx = min(px0 + 90, WORLD_W - 4)
            if in_bounds(spine_y, spx) and not plants[spine_y, spx]:
                return (Tool.COALPOWER, spx, spine_y)

    # ---- Phase 3: Vertical wire branches from spine ----
    branch_spacing = 4
    branch_len = 16

    for bx in range(px0 + 4, spine_end_x, branch_spacing):
        if not in_bounds(spine_y, bx):
            continue
        if not (wires[spine_y, bx] or plants[spine_y, bx]):
            continue
        # Upward branch
        for vy in range(spine_y - 1, max(spine_y - branch_len, 1), -1):
            if not in_bounds(vy, bx):
                break
            if emp[vy, bx]:
                if power_grid[vy + 1, bx]:
                    return (Tool.WIRE, bx, vy)
                break
            elif not power_grid[vy, bx]:
                break
        # Downward branch
        for vy in range(spine_y + 1, min(spine_y + branch_len, WORLD_H - 2)):
            if not in_bounds(vy, bx):
                break
            if emp[vy, bx]:
                if power_grid[vy - 1, bx]:
                    return (Tool.WIRE, bx, vy)
                break
            elif not power_grid[vy, bx]:
                break

    # ---- Phase 4: Horizontal roads parallel to spine ----
    road_offsets = [-3, 3, -7, 7, -11, 11, -15, 15]
    for offset in road_offsets:
        ry = spine_y + offset
        if ry < 2 or ry >= WORLD_H - 2:
            continue
        road_row = roads[ry, :]
        if np.any(road_row):
            rightmost_road = int(np.max(np.where(road_row)[0]))
        else:
            rightmost_road = px0 + 1

        for rx in range(rightmost_road + 1, spine_end_x + 5):
            if not in_bounds(ry, rx):
                break
            if roads[ry, rx]:
                rightmost_road = rx
                continue
            if not emp[ry, rx]:
                break
            if near_mask(power_grid, rx, ry, 6):
                return (Tool.ROAD, rx, ry)

    # ---- Phase 5: Vertical connector roads at branch points ----
    for cx in range(px0 + 4, spine_end_x + 1, branch_spacing):
        for cy in range(spine_y - 14, spine_y + 15):
            if cy < 0 or cy >= WORLD_H:
                continue
            if emp[cy, cx] and near_mask(roads, cx, cy, 1):
                if near_mask(power_grid, cx, cy, 3):
                    return (Tool.ROAD, cx, cy)

    # ---- Phase 6: Zone placement ----
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))

    # RES:COM:IND roughly 4:2:2
    if total_res == 0:
        desired = Tool.RESIDENTIAL
    elif total_ind == 0:
        desired = Tool.INDUSTRIAL
    elif total_com == 0:
        desired = Tool.COMMERCIAL
    elif total_ind * 4 < total_res:
        desired = Tool.INDUSTRIAL
    elif total_com * 4 < total_res:
        desired = Tool.COMMERCIAL
    else:
        desired = Tool.RESIDENTIAL

    zone_any = res | com | ind
    candidates = []
    center_x = (px0 + spine_end_x) // 2

    for zy in range(2, WORLD_H - 2):
        for zx in range(2, WORLD_W - 2):
            if not emp[zy, zx]:
                continue
            if zy == spine_y:
                continue
            if not near_mask(roads, zx, zy, 2):
                continue
            if not near_mask(power_grid, zx, zy, 5):
                continue
            # Check 3x3 footprint is mostly empty
            y0, y1 = max(0, zy - 1), min(WORLD_H, zy + 2)
            x0, x1 = max(0, zx - 1), min(WORLD_W, zx + 2)
            region = emp[y0:y1, x0:x1]
            if region.size < 9 or int(np.sum(region)) < 7:
                continue
            near_zone = near_mask(zone_any, zx, zy, 3)
            near_road_adj = near_mask(roads, zx, zy, 1)
            road_count = count_mask(roads, zx, zy, 2)
            power_count = count_mask(power_grid, zx, zy, 4)
            dist_center = abs(zx - center_x) + abs(zy - spine_y) * 1.5
            score = (dist_center
                     - (8 if near_zone else 0)
                     - (12 if near_road_adj else 0)
                     - road_count * 2
                     - power_count)
            candidates.append((score, zx, zy))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        top_n = max(1, min(20, len(candidates)))
        # Use both step and city_pop for selection variety
        idx = (step + obs.city_pop * 3) % top_n
        _, zx, zy = candidates[idx]
        return (desired, zx, zy)

    # ---- Phase 7: Extend power grid to reach new empty areas ----
    pg_adj = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    pg_adj[1:, :] |= power_grid[:-1, :]
    pg_adj[:-1, :] |= power_grid[1:, :]
    pg_adj[:, 1:] |= power_grid[:, :-1]
    pg_adj[:, :-1] |= power_grid[:, 1:]
    pg_frontier = pg_adj & emp & ~roads
    wf_ys, wf_xs = np.where(pg_frontier)
    if len(wf_xs) > 0:
        dists = np.abs(wf_xs - center_x) + np.abs(wf_ys - spine_y)
        idx = int(np.argmin(dists))
        return (Tool.WIRE, int(wf_xs[idx]), int(wf_ys[idx]))

    # ---- Phase 8: Extend roads into powered empty areas ----
    rd_adj = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    rd_adj[1:, :] |= roads[:-1, :]
    rd_adj[:-1, :] |= roads[1:, :]
    rd_adj[:, 1:] |= roads[:, :-1]
    rd_adj[:, :-1] |= roads[:, 1:]
    rd_frontier = rd_adj & emp
    rf_ys, rf_xs = np.where(rd_frontier)
    if len(rf_xs) > 0:
        powered_front = []
        for i in range(len(rf_xs)):
            rx2, ry2 = int(rf_xs[i]), int(rf_ys[i])
            if near_mask(power_grid, rx2, ry2, 5):
                sc = abs(rx2 - center_x) + abs(ry2 - spine_y)
                powered_front.append((sc, rx2, ry2))
        if powered_front:
            powered_front.sort()
            _, rx2, ry2 = powered_front[0]
            return (Tool.ROAD, rx2, ry2)

    return None
```

## best open-loop
- cityPop=2540 (R=143 C=0 I=0) | stored cityPop=3027 | origin=crossover iter=7515
- reactivity: cf_sensitivity=0.00, traj_divergence=0.00, ind_share=0.00  [reactive-map]
- code: 218 lines, 7291 chars | motifs: precomputed-plan, reads-map, plants=3c, wire, road
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
...........RRRRRRRR.........................................
...........RRRRRRRR.........................................
...........=========........................................
...........=..RRRRR=........................................
...........=*.RRRRR=........................................
...........=..RRRRR=........................................
...........===RRRRR=........................................
...........RRRRRRRR.........................................
...........RRRRRRRR.........................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
..........................................*.................
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
def build_plan():
    plan = []
    
    # --- Neighborhood 1: Left cluster centered at (30, 35) ---
    cx1, cy1 = 30, 35
    
    # Coal power plant for left cluster (offset so wire can connect)
    plan.append((Tool.COALPOWER, cx1 - 5, cy1))
    
    # Connect power plant to vertical wire spine with horizontal wires
    for dx in range(-4, 0):
        plan.append((Tool.WIRE, cx1 + dx, cy1))
    
    # Vertical wire spine at cx1
    for dy in range(-10, 11):
        plan.append((Tool.WIRE, cx1, cy1 + dy))
    
    # Horizontal road above and below center
    for dx in range(-8, 9):
        plan.append((Tool.ROAD, cx1 + dx, cy1 - 4))
        plan.append((Tool.ROAD, cx1 + dx, cy1 + 4))
    
    # Vertical roads at edges to connect horizontal roads
    for dy in range(-4, 5):
        plan.append((Tool.ROAD, cx1 - 8, cy1 + dy))
        plan.append((Tool.ROAD, cx1 + 8, cy1 + dy))
    
    # Residential zones between roads (above)
    for dx in [-6, -3, 0, 3, 6]:
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 - 7))
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 - 2))
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 + 2))
        plan.append((Tool.RESIDENTIAL, cx1 + dx, cy1 + 7))
    
    # Commercial near center
    for dy in [-2, 2]:
        plan.append((Tool.COMMERCIAL, cx1, cy1 + dy))
    
    # Industrial at edges
    plan.append((Tool.INDUSTRIAL, cx1 - 6, cy1))
    plan.append((Tool.INDUSTRIAL, cx1 + 6, cy1))
    
    # Fire/Police
    plan.append((Tool.FIRESTATION, cx1 - 8, cy1 - 10))
    plan.append((Tool.POLICESTATION, cx1 + 8, cy1 - 10))
    
    # --- Neighborhood 2: Right cluster centered at (80, 60) ---
    cx2, cy2 = 80, 60
    
    # Coal power plant for right cluster
    plan.append((Tool.COALPOWER, cx2 + 5, cy2))
    
    # Connect power plant to vertical wire spine
    for dx in range(1, 5):
        plan.append((Tool.WIRE, cx2 + dx, cy2))
    
    # Vertical wire spine at cx2
    for dy in range(-10, 11):
        plan.append((Tool.WIRE, cx2, cy2 + dy))
    
    # Horizontal roads above and below center
    for dx in range(-8, 9):
        plan.append((Tool.ROAD, cx2 + dx, cy2 - 4))
        plan.append((Tool.ROAD, cx2 + dx, cy2 + 4))
    
    # Vertical roads at edges
    for dy in range(-4, 5):
        plan.append((Tool.ROAD, cx2 - 8, cy2 + dy))
        plan.append((Tool.ROAD, cx2 + 8, cy2 + dy))
    
    # Residential zones
    for dx in [-6, -3, 0, 3, 6]:
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 - 7))
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 - 2))
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 + 2))
        plan.append((Tool.RESIDENTIAL, cx2 + dx, cy2 + 7))
    
    # Commercial
    for dy in [-2, 2]:
        plan.append((Tool.COMMERCIAL, cx2, cy2 + dy))
    for dx in [-6, 6]:
        plan.append((Tool.COMMERCIAL, cx2 + dx, cy2))
    
    # Industrial
    plan.append((Tool.INDUSTRIAL, cx2 - 3, cy2))
    plan.append((Tool.INDUSTRIAL, cx2 + 3, cy2))
    
    # Fire/Police
    plan.append((Tool.FIRESTATION, cx2 - 8, cy2 + 10))
    plan.append((Tool.POLICESTATION, cx2 + 8, cy2 + 10))
    
    # --- Middle cluster at (55, 47) ---
    cx3, cy3 = 55, 47
    plan.append((Tool.COALPOWER, cx3, cy3 - 12))
    for dy in range(-11, 0):
        plan.append((Tool.WIRE, cx3, cy3 + dy))
    for dy in range(0, 10):
        plan.append((Tool.WIRE, cx3, cy3 + dy))
    for dx in range(-7, 8):
        plan.append((Tool.ROAD, cx3 + dx, cy3 - 4))
        plan.append((Tool.ROAD, cx3 + dx, cy3 + 4))
    for dy in range(-4, 5):
        plan.append((Tool.ROAD, cx3 - 7, cy3 + dy))
        plan.append((Tool.ROAD, cx3 + 7, cy3 + dy))
    for dx in [-6, -3, 0, 3, 6]:
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 - 7))
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 - 2))
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 + 2))
        plan.append((Tool.RESIDENTIAL, cx3 + dx, cy3 + 7))
    plan.append((Tool.COMMERCIAL, cx3 - 3, cy3))
    plan.append((Tool.COMMERCIAL, cx3 + 3, cy3))
    plan.append((Tool.INDUSTRIAL, cx3, cy3))
    
    # --- Connecting roads between clusters ---
    # Left to middle
    for x in range(cx1 + 8, cx3 - 7 + 1):
        plan.append((Tool.ROAD, x, cy1 + 4))
    # Connect vertically from left road to middle road height
    for y in range(min(cy1 + 4, cy3 - 4), max(cy1 + 4, cy3 - 4) + 1):
        plan.append((Tool.ROAD, cx1 + 8, y))
    
    # Middle to right
    for x in range(cx3 + 7, cx2 - 8 + 1):
        plan.append((Tool.ROAD, x, cy2 - 4))
    for y in range(min(cy3 + 4, cy2 - 4), max(cy3 + 4, cy2 - 4) + 1):
        plan.append((Tool.ROAD, cx2 - 8, y))
    
    # Horizontal wire connecting power plants (for redundancy)
    for x in range(cx1 - 4, cx3 + 1):
        plan.append((Tool.WIRE, x, cy1))
    
    return plan


def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
        state["phase"] = 1
    
    tm = obs.tile_map
    emp = empty_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    roads = road_mask(tm)
    wires = wire_mask(tm)
    plants = plant_mask(tm)
    power_grid = plants | wires
    
    if state["phase"] == 1:
        i = state["i"]
        plan = state["plan"]
        if i < len(plan):
            state["i"] = i + 1
            tool, x, y = plan[i]
            # Clamp to valid range
            x = max(1, min(WORLD_W - 2, x))
            y = max(1, min(WORLD_H - 2, y))
            return (tool, x, y)
        else:
            state["phase"] = 2
    
    # Phase 2: reactive gap-filling
    # Find empty cells near roads AND near power grid
    def dilate(mask, dist=3):
        result = mask.copy()
        for shift in range(1, dist + 1):
            result[shift:, :] |= mask[:-shift, :]
            result[:-shift, :] |= mask[shift:, :]
            result[:, shift:] |= mask[:, :-shift]
            result[:, :-shift] |= mask[:, shift:]
        return result
    
    road_near = dilate(roads, 3)
    wire_near = dilate(power_grid, 5)
    existing = res | com | ind
    zone_near = dilate(existing, 4)
    
    # Primary candidates: near road, wire, and existing zones
    cand = emp & road_near & wire_near & zone_near
    ys, xs = np.where(cand)
    
    if len(xs) == 0:
        cand2 = emp & road_near & wire_near
        ys, xs = np.where(cand2)
        if len(xs) == 0:
            # Try extending power grid
            pg_adj = dilate(power_grid, 1) & emp
            pys, pxs = np.where(pg_adj)
            if len(pxs) > 0:
                k = int(obs.step) % len(pxs)
                return (Tool.WIRE, int(pxs[k]), int(pys[k]))
            return None
    
    # Deterministic but varied selection
    k = int(obs.step) % len(xs)
    x, y = int(xs[k]), int(ys[k])
    
    # Balance zone types based on current counts
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))
    total = total_res + total_com + total_ind + 1
    
    # Target ratio: 5 res : 3 com : 2 ind
    res_frac = total_res / total
    com_frac = total_com / total
    ind_frac = total_ind / total
    
    if ind_frac < 0.15:
        tool = Tool.INDUSTRIAL
    elif com_frac < 0.25:
        tool = Tool.COMMERCIAL
    else:
        tool = Tool.RESIDENTIAL
    
    return (tool, x, y)
```

## most industrial
- cityPop=320 (R=0 C=0 I=3) | stored cityPop=640 | origin=mutate iter=3210
- reactivity: cf_sensitivity=0.95, traj_divergence=0.06, ind_share=1.00  [reactive-map]
- code: 177 lines, 6392 chars | motifs: reads-map, reads-stats, plants=3c, wire, road
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
.......II=======............................................
.......II..=.II.............................................
........IIII.II.............................................
........IIII====............................................
........IIIIII..............................................
........IIIIII..............................................
.......*..II..................*................*............
..........II.II.............................................
.......II====II=............................................
.......II..=II..............................................
.........II=II..............................................
.........II=====............................................
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
    tm = obs.tile_map
    emp = empty_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    wires = wire_mask(tm)
    roads = road_mask(tm)
    plants = plant_mask(tm)

    step = int(obs.step)
    power_grid = plants | wires

    def in_bounds(y, x):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    def near_mask(mask, x, y, dist=1):
        y0 = max(0, y - dist); y1 = min(WORLD_H, y + dist + 1)
        x0 = max(0, x - dist); x1 = min(WORLD_W, x + dist + 1)
        return np.any(mask[y0:y1, x0:x1])

    plant_ys, plant_xs = np.where(plants)
    n_plants = len(plant_xs)

    # Place first power plant - left side
    if n_plants == 0:
        return (Tool.COALPOWER, 15, 50)

    px, py = int(plant_xs[0]), int(plant_ys[0])

    # Two horizontal wire spines: one above, one below the plant row
    # This creates two separate neighborhoods for variety
    spine_y1 = py - 8   # upper spine
    spine_y2 = py + 8   # lower spine
    spine_start_x = px + 3
    spine_end_x = min(px + 90, WORLD_W - 3)

    # Build wire from plant along py row first (connector)
    # Then branch up/down
    connector_row = wires[py, :] | plants[py, :]
    rightmost_connector = px
    if np.any(connector_row):
        rightmost_connector = int(np.max(np.where(connector_row)[0]))

    if rightmost_connector < spine_end_x:
        nx = rightmost_connector + 1
        if in_bounds(py, nx) and emp[py, nx]:
            if near_mask(power_grid, nx, py, 1):
                return (Tool.WIRE, nx, py)

    # Second coal plant for upper neighborhood
    if n_plants == 1 and step > 20 and obs.city_pop > 10:
        mid_x = px + 45
        if in_bounds(py, mid_x) and emp[py, mid_x]:
            return (Tool.COALPOWER, mid_x, py)

    # Third coal plant further right
    if n_plants == 2 and step > 60 and obs.city_pop > 80:
        mid_x = px + 80
        if in_bounds(py, mid_x) and emp[py, mid_x]:
            return (Tool.COALPOWER, mid_x, py)

    # Vertical wire branches: go up to spine_y1 and down to spine_y2
    # Do this at regular intervals along the connector wire
    for bx in range(spine_start_x, spine_end_x, 6):
        if not (wires[py, bx] or plants[py, bx]):
            continue
        # Branch up to spine_y1
        for vy in range(py - 1, max(spine_y1 - 1, 0), -1):
            if emp[vy, bx] and near_mask(power_grid, bx, vy, 1):
                return (Tool.WIRE, bx, vy)
            elif not emp[vy, bx] and not power_grid[vy, bx]:
                break
        # Branch down to spine_y2
        for vy in range(py + 1, min(spine_y2 + 1, WORLD_H - 1)):
            if emp[vy, bx] and near_mask(power_grid, bx, vy, 1):
                return (Tool.WIRE, bx, vy)
            elif not emp[vy, bx] and not power_grid[vy, bx]:
                break

    # Road rows: two per neighborhood
    # Upper neighborhood: roads at spine_y1-3 and spine_y1+3
    # Lower neighborhood: roads at spine_y2-3 and spine_y2+3
    road_rows = []
    for sy in [spine_y1, spine_y2]:
        road_rows.append(sy - 3)
        road_rows.append(sy + 3)

    for ry in road_rows:
        if ry < 2 or ry >= WORLD_H - 2:
            continue
        row_roads = roads[ry, :]
        if np.any(row_roads):
            rightmost_road = int(np.max(np.where(row_roads)[0]))
        else:
            rightmost_road = spine_start_x - 1

        if rightmost_road < spine_end_x:
            rx = rightmost_road + 1
            if not np.any(row_roads):
                rx = spine_start_x
            if in_bounds(ry, rx) and near_mask(power_grid, rx, ry, 15):
                return (Tool.ROAD, rx, ry)

    # Vertical connector roads every 10 tiles
    for cx in range(spine_start_x + 5, spine_end_x, 10):
        all_road_ys = sorted([r for r in road_rows if 2 <= r < WORLD_H - 2])
        if len(all_road_ys) < 2:
            continue
        for cy in range(all_road_ys[0], all_road_ys[-1] + 1):
            if cy < 2 or cy >= WORLD_H - 2:
                continue
            if emp[cy, cx] and near_mask(roads, cx, cy, 1):
                return (Tool.ROAD, cx, cy)

    # Zone placement: heavy industrial/commercial bias
    # Target ratio: 1 RES : 2 COM : 3 IND
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))

    if total_ind < total_res * 3 or total_res == 0:
        desired = Tool.INDUSTRIAL
    elif total_com < total_res * 2 or total_res == 0:
        desired = Tool.COMMERCIAL
    else:
        desired = Tool.RESIDENTIAL

    zone_any = res | com | ind
    candidates = []

    center_x = px + 45

    for zy in range(2, WORLD_H - 2):
        # Only place zones near our neighborhood rows
        near_any_road_row = any(abs(zy - ry) <= 4 for ry in road_rows if 2 <= ry < WORLD_H - 2)
        if not near_any_road_row:
            continue
        for zx in range(2, WORLD_W - 2):
            if not emp[zy, zx]:
                continue
            if not near_mask(roads, zx, zy, 2):
                continue
            if not near_mask(power_grid, zx, zy, 12):
                continue
            y0, y1 = max(0, zy - 1), min(WORLD_H, zy + 2)
            x0, x1 = max(0, zx - 1), min(WORLD_W, zx + 2)
            region = emp[y0:y1, x0:x1]
            if region.size < 9 or int(np.sum(region)) < 7:
                continue
            near_road = near_mask(roads, zx, zy, 1)
            near_zone = near_mask(zone_any, zx, zy, 4)
            dist_center = abs(zx - center_x) + abs(zy - py) * 0.5
            score = dist_center - (8 if near_zone else 0) - (12 if near_road else 0)
            candidates.append((score, zx, zy))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        top_n = max(1, min(60, len(candidates)))
        idx = step % top_n
        _, zx, zy = candidates[idx]
        return (desired, zx, zy)

    # Fallback: extend wire toward zone cluster center
    pg_adj = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    pg_adj[1:, :]  |= power_grid[:-1, :]
    pg_adj[:-1, :] |= power_grid[1:, :]
    pg_adj[:, 1:]  |= power_grid[:, :-1]
    pg_adj[:, :-1] |= power_grid[:, 1:]
    pg_frontier = pg_adj & emp
    wf_ys, wf_xs = np.where(pg_frontier)
    if len(wf_xs) > 0:
        dists = np.abs(wf_xs - center_x) + np.abs(wf_ys - py)
        idx = int(np.argmin(dists))
        return (Tool.WIRE, int(wf_xs[idx]), int(wf_ys[idx]))

    return None
```

