# Deep analysis — elm_clind_long
archive: 417 cells, qd=999798, best cityPop=14908

## champion (max cityPop)
- cityPop=15380 (R=804 C=0 I=0) | stored cityPop=14908 | origin=mutate+cl iter=9308
- reactivity: cf_sensitivity=0.65, traj_divergence=0.00, ind_share=0.00  [reactive-map]
- code: 380 lines, 13882 chars | motifs: reads-map, reads-stats, plants=3c, wire, road
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
...................................*........................
............................................................
............................................................
............................................................
...................RR.......................................
..................RRRR......................................
..................RRRR......................................
..................RRRR......................................
...............RRRRRRRRRRRRR................................
............*.RRRRRRRRRRRRRRRR..............................
..............RRRRRRRRRRRRRRRR..............................
...............RRRRRRRRRRRRR................................
..................RRRR......................................
..................RRRRR.....................................
..................RRRRR.....................................
...................RR.......................................
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
    wir = wire_mask(tm)
    roa = road_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    plt = plant_mask(tm)
    pop = obs.city_pop

    MARGIN = 4

    def in_bounds(x, y):
        return MARGIN <= x < WORLD_W - MARGIN and MARGIN <= y < WORLD_H - MARGIN

    def dilate(mask):
        d = mask.copy()
        d[1:, :] |= mask[:-1, :]
        d[:-1, :] |= mask[1:, :]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        return d

    def dilate_n(mask, n):
        d = mask.copy()
        for _ in range(n):
            d = dilate(d)
        return d

    margin_mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    margin_mask[MARGIN:WORLD_H-MARGIN, MARGIN:WORLD_W-MARGIN] = True

    power_src = plt | wir
    zones_any = res | com | ind

    def power_flood(src, steps=80):
        covered = src.copy()
        traversable = wir | plt | res | com | ind
        for _ in range(steps):
            nxt = dilate(covered) & traversable
            if not np.any(nxt & ~covered):
                break
            covered = nxt | src
        return covered

    power_cov = power_flood(power_src)

    plant_ys, plant_xs = np.where(plt)
    px_all = plant_xs.tolist()
    py_all = plant_ys.tolist()

    n_res = int(res.sum())
    n_com = int(com.sum())
    n_ind = int(ind.sum())
    total_zones = max(1, n_res + n_com + n_ind)
    r_f = n_res / total_zones
    c_f = n_com / total_zones
    i_f = n_ind / total_zones

    # Zone mix based on current population
    if pop < 50:
        ztool = Tool.RESIDENTIAL
    elif pop < 300:
        if c_f <= 0.25:
            ztool = Tool.COMMERCIAL
        elif i_f <= 0.20:
            ztool = Tool.INDUSTRIAL
        else:
            ztool = Tool.RESIDENTIAL
    elif pop < 800:
        if r_f <= 0.40:
            ztool = Tool.RESIDENTIAL
        elif c_f <= 0.35:
            ztool = Tool.COMMERCIAL
        elif i_f <= 0.25:
            ztool = Tool.INDUSTRIAL
        else:
            ztool = Tool.RESIDENTIAL
    else:
        if r_f <= 0.45:
            ztool = Tool.RESIDENTIAL
        elif c_f <= 0.30:
            ztool = Tool.COMMERCIAL
        elif i_f <= 0.20:
            ztool = Tool.INDUSTRIAL
        else:
            ztool = Tool.RESIDENTIAL

    def can_place_3x3(cx, cy):
        if not in_bounds(cx, cy):
            return False
        y0, y1 = cy - 1, cy + 2
        x0, x1 = cx - 1, cx + 2
        if y0 < 0 or y1 > WORLD_H or x0 < 0 or x1 > WORLD_W:
            return False
        return np.all(emp[y0:y1, x0:x1])

    # --- CLOSED-LOOP REACTIVE LAYOUT ---
    # Use a horizontal road spine at row ~50 (center of map)
    # with vertical branches. Power plants placed at left side.
    # All decisions based on what currently exists on the map.

    SPINE_ROW = 50  # horizontal road spine
    WIRE_ROW = 48   # horizontal wire row (parallel, above spine)

    # How many clusters to activate based on current population
    n_active = 1
    if pop > 100:
        n_active = 2
    if pop > 400:
        n_active = 3
    if pop > 900:
        n_active = 4

    # Clusters arranged horizontally along the spine
    # Each cluster: center_x for vertical branch
    clusters = [
        {'cx': 40, 'cy': SPINE_ROW, 'plant_x': 25, 'plant_y': 48,
         'branch_x': 40, 'x0': 32, 'x1': 55, 'y0': 42, 'y1': 58},
        {'cx': 65, 'cy': SPINE_ROW, 'plant_x': 25, 'plant_y': 48,
         'branch_x': 65, 'x0': 57, 'x1': 80, 'y0': 42, 'y1': 58},
        {'cx': 90, 'cy': SPINE_ROW, 'plant_x': 25, 'plant_y': 48,
         'branch_x': 90, 'x0': 82, 'x1': 105, 'y0': 42, 'y1': 58},
        {'cx': 15, 'cy': SPINE_ROW, 'plant_x': 10, 'plant_y': 48,
         'branch_x': 15, 'x0': 7, 'x1': 30, 'y0': 42, 'y1': 58},
    ]
    active_clusters = clusters[:n_active]

    # ---- Phase 0: Ensure first power plant ----
    if not np.any(plt):
        cl = clusters[0]
        return (Tool.COALPOWER, cl['plant_x'], cl['plant_y'])

    # ---- Phase 1: Ensure each active cluster has a power plant nearby ----
    for cl in active_clusters:
        has_plant = any(
            abs(px - cl['plant_x']) < 40 and abs(py - cl['plant_y']) < 15
            for px, py in zip(px_all, py_all)
        )
        if not has_plant:
            px, py = cl['plant_x'], cl['plant_y']
            # Find nearest empty spot
            for r in range(20):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        nx, ny = px + dx, py + dy
                        if in_bounds(nx, ny) and emp[ny, nx]:
                            return (Tool.COALPOWER, nx, ny)

    # Add extra plants for large populations
    n_plants = len(px_all)
    if pop > 800 and n_plants < n_active + 1:
        epx, epy = 70, 30
        local_has = any(abs(px - epx) < 20 and abs(py - epy) < 20
                       for px, py in zip(px_all, py_all))
        if not local_has:
            for r in range(15):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        nx, ny = epx + dx, epy + dy
                        if in_bounds(nx, ny) and emp[ny, nx]:
                            return (Tool.COALPOWER, nx, ny)

    # ---- Phase 2: Build horizontal wire row (WIRE_ROW) from plant outward ----
    # Wire connects power plants to zones reactively
    # Find where wire already exists and extend it
    for cl in active_clusters:
        x0 = max(MARGIN + 1, cl['x0'])
        x1 = min(WORLD_W - MARGIN - 1, cl['x1'])

        # Check if wire row has any power source in this cluster's range
        wire_in_range = np.any(power_src[WIRE_ROW, x0:x1+1])

        if not wire_in_range:
            # Bootstrap: extend wire from nearest power source toward cluster
            wf = dilate(power_src) & emp & margin_mask
            if np.any(wf):
                wfy, wfx = np.where(wf)
                # Target: wire row at cluster branch_x
                target_x = cl['branch_x']
                dists = np.abs(wfx - target_x) + np.abs(wfy - WIRE_ROW) * 2
                best = int(np.argmin(dists))
                bx, by = int(wfx[best]), int(wfy[best])
                if in_bounds(bx, by):
                    return (Tool.WIRE, bx, by)

        # Extend wire horizontally along WIRE_ROW within cluster range
        # from existing power source
        for wx in list(range(cl['branch_x'], x0 - 1, -1)) + list(range(cl['branch_x'] + 1, x1 + 1)):
            if not in_bounds(wx, WIRE_ROW):
                continue
            if power_src[WIRE_ROW, wx]:
                continue
            if not emp[WIRE_ROW, wx]:
                continue
            adj = any(
                0 <= WIRE_ROW + dr < WORLD_H and 0 <= wx + dc < WORLD_W
                and power_src[WIRE_ROW + dr, wx + dc]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if adj:
                return (Tool.WIRE, wx, WIRE_ROW)

    # ---- Phase 3: Connect wire rows between clusters ----
    if n_active >= 2:
        # Extend wire horizontally between cluster ranges
        all_x0 = min(cl['x0'] for cl in active_clusters)
        all_x1 = max(cl['x1'] for cl in active_clusters)
        for wx in range(max(MARGIN + 1, all_x0), min(WORLD_W - MARGIN - 1, all_x1) + 1):
            if not in_bounds(wx, WIRE_ROW):
                continue
            if power_src[WIRE_ROW, wx]:
                continue
            if not emp[WIRE_ROW, wx]:
                continue
            adj = any(
                0 <= WIRE_ROW + dr < WORLD_H and 0 <= wx + dc < WORLD_W
                and power_src[WIRE_ROW + dr, wx + dc]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if adj:
                return (Tool.WIRE, wx, WIRE_ROW)

    # ---- Phase 4: Build vertical wire branches from WIRE_ROW ----
    for cl in active_clusters:
        bx = cl['branch_x']
        y0 = max(MARGIN + 1, cl['y0'])
        y1 = min(WORLD_H - MARGIN - 1, cl['y1'])

        if not in_bounds(bx, WIRE_ROW) or not power_src[WIRE_ROW, bx]:
            continue

        # Extend vertical wire up and down from WIRE_ROW
        for wy in list(range(WIRE_ROW - 1, y0 - 1, -1)) + list(range(WIRE_ROW + 1, y1 + 1)):
            if not in_bounds(bx, wy):
                continue
            if power_src[wy, bx]:
                continue
            if not emp[wy, bx]:
                continue
            adj = any(
                0 <= wy + dr < WORLD_H and 0 <= bx + dc < WORLD_W
                and power_src[wy + dr, bx + dc]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if adj:
                return (Tool.WIRE, bx, wy)

    # ---- Phase 5: Build horizontal road spine (SPINE_ROW) ----
    if not np.any(roa):
        cl = clusters[0]
        if in_bounds(cl['branch_x'], SPINE_ROW) and emp[SPINE_ROW, cl['branch_x']]:
            return (Tool.ROAD, cl['branch_x'], SPINE_ROW)

    road_front = dilate(roa) & emp & margin_mask

    for cl in active_clusters:
        x0 = max(MARGIN + 1, cl['x0'])
        x1 = min(WORLD_W - MARGIN - 1, cl['x1'])

        # Extend road along SPINE_ROW
        for rx in list(range(cl['branch_x'], x0 - 1, -1)) + list(range(cl['branch_x'] + 1, x1 + 1)):
            if not in_bounds(rx, SPINE_ROW):
                continue
            if road_front[SPINE_ROW, rx] and emp[SPINE_ROW, rx]:
                return (Tool.ROAD, rx, SPINE_ROW)

    # Connect road between clusters
    if n_active >= 2:
        all_x0 = min(cl['x0'] for cl in active_clusters)
        all_x1 = max(cl['x1'] for cl in active_clusters)
        for rx in range(max(MARGIN + 1, all_x0), min(WORLD_W - MARGIN - 1, all_x1) + 1):
            if not in_bounds(rx, SPINE_ROW):
                continue
            if road_front[SPINE_ROW, rx] and emp[SPINE_ROW, rx]:
                return (Tool.ROAD, rx, SPINE_ROW)

    # ---- Phase 6: Build vertical road branches from SPINE_ROW ----
    for cl in active_clusters:
        bx = cl['branch_x']
        y0 = max(MARGIN + 1, cl['y0'])
        y1 = min(WORLD_H - MARGIN - 1, cl['y1'])

        if not in_bounds(bx, SPINE_ROW) or not roa[SPINE_ROW, bx]:
            # Try to get road to this branch_x first
            if road_front[SPINE_ROW, bx] and emp[SPINE_ROW, bx] and in_bounds(bx, SPINE_ROW):
                return (Tool.ROAD, bx, SPINE_ROW)
            continue

        road_front2 = dilate(roa) & emp & margin_mask
        for ry in list(range(SPINE_ROW - 1, y0 - 1, -1)) + list(range(SPINE_ROW + 1, y1 + 1)):
            if not in_bounds(bx, ry):
                continue
            if road_front2[ry, bx] and emp[ry, bx]:
                return (Tool.ROAD, bx, ry)

    # ---- Phase 7: Wire unpowered zones reactively ----
    uncov = zones_any & ~power_cov
    if np.any(uncov):
        wf = dilate(power_src) & emp & margin_mask
        if np.any(wf):
            zy, zx = np.where(uncov)
            wfy2, wfx2 = np.where(wf)
            best_d, best_w = 1e9, None
            for i in range(min(len(wfx2), 300)):
                d = int(np.min(np.abs(wfx2[i] - zx) + np.abs(wfy2[i] - zy)))
                if d < best_d:
                    best_d = d
                    best_w = (int(wfx2[i]), int(wfy2[i]))
            if best_w and in_bounds(best_w[0], best_w[1]):
                return (Tool.WIRE, best_w[0], best_w[1])

    # ---- Phase 8: Place zones reactively ----
    # Re-compute power coverage
    power_cov = power_flood(power_src)
    near_road = dilate_n(roa, 3)

    road_targets = [(cl['branch_x'], SPINE_ROW) for cl in active_clusters]

    candidates = emp & near_road & power_cov & margin_mask
    if not np.any(candidates):
        candidates = emp & near_road & margin_mask

    if np.any(candidates):
        ys, xs = np.where(candidates)
        near_zones = dilate(zones_any)

        dists = np.array([
            min(abs(int(xs[i]) - rx) + abs(int(ys[i]) - ry) for rx, ry in road_targets)
            for i in range(len(xs))
        ])
        power_pen = np.where(power_cov[ys, xs], 0, 20)
        cluster_bonus = np.where(near_zones[ys, xs], -4, 5)
        scores = dists + power_pen + cluster_bonus
        sorted_idxs = np.argsort(scores)

        for idx in sorted_idxs[:400]:
            ty2, tx2 = int(ys[idx]), int(xs[idx])
            if can_place_3x3(tx2, ty2):
                return (ztool, tx2, ty2)

        for idx in sorted_idxs[:100]:
            ty2, tx2 = int(ys[idx]), int(xs[idx])
            if in_bounds(tx2, ty2):
                return (ztool, tx2, ty2)

    # ---- Phase 9: Extend wire frontier toward best target ----
    wf = dilate(power_src) & emp & margin_mask
    if np.any(wf):
        wfy2, wfx2 = np.where(wf)
        best_i, best_d = 0, 1e9
        for i in range(len(wfx2)):
            x2, y2 = int(wfx2[i]), int(wfy2[i])
            d = min(abs(x2 - cl['branch_x']) + abs(y2 - WIRE_ROW)
                    for cl in active_clusters)
            if d < best_d:
                best_d = d
                best_i = i
        return (Tool.WIRE, int(wfx2[best_i]), int(wfy2[best_i]))

    # ---- Phase 10: Extend road network ----
    rf2 = dilate(roa) & emp & margin_mask
    if np.any(rf2):
        ry_arr, rx_arr = np.where(rf2)
        dists = np.array([
            min(abs(int(rx_arr[i]) - cl['branch_x']) + abs(int(ry_arr[i]) - SPINE_ROW)
                for cl in active_clusters)
            for i in range(len(rx_arr))
        ])
        idx = int(np.argmin(dists))
        x2, y2 = int(rx_arr[idx]), int(ry_arr[idx])
        if in_bounds(x2, y2):
            return (Tool.ROAD, x2, y2)

    return None
```

## most counterfactual-sensitive (max cf)
- cityPop=9760 (R=488 C=0 I=0) | stored cityPop=9408 | origin=crossover iter=7932
- reactivity: cf_sensitivity=1.00, traj_divergence=0.00, ind_share=0.00  [reactive-map]
- code: 295 lines, 11765 chars | motifs: reads-map, reads-stats, plants=2c, wire, road
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
..............................*.............................
.........................RRRRR.RRRR.........................
.......................RRRRRRR=RRRRRR.......................
.......................RRRRRRRRRRRRRR.......................
.........................RRRRRRRRRRR........................
............................................................
.....=========================================..............
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
    wir = wire_mask(tm)
    roa = road_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    plt = plant_mask(tm)

    MARGIN = 3
    pop = obs.city_pop

    def in_bounds(x, y):
        return MARGIN <= x < WORLD_W - MARGIN and MARGIN <= y < WORLD_H - MARGIN

    def dilate(mask):
        d = mask.copy()
        d[1:, :] |= mask[:-1, :]
        d[:-1, :] |= mask[1:, :]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        return d

    def dilate_n(mask, n):
        d = mask.copy()
        for _ in range(n):
            d = dilate(d)
        return d

    def power_coverage():
        src = plt | wir
        traversable = wir | plt | roa | res | com | ind
        covered = src.copy()
        for _ in range(80):
            prev = covered.copy()
            covered = (dilate(covered) & traversable) | src
            if np.array_equal(covered, prev):
                break
        return covered

    margin_mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    margin_mask[MARGIN:WORLD_H - MARGIN, MARGIN:WORLD_W - MARGIN] = True

    power_src = plt | wir
    zones_any = res | com | ind

    # ---- Phase 0: Place first coal power plant ----
    if not np.any(plt):
        # Place near top-center for maximum downward expansion room
        return (Tool.COALPOWER, 60, 18)

    plant_ys, plant_xs = np.where(plt)
    plant_cx = int(np.mean(plant_xs))
    plant_cy = int(np.mean(plant_ys))
    n_plants = max(1, int(np.sum(plt)) // 4)

    power_cov = power_coverage()

    # Vertical wire spine at plant_cx
    spine_x = plant_cx

    # Corridor rows below the plant (horizontal roads every 8 rows)
    corridor_start_y = plant_cy + 5
    corridor_rows = []
    for i in range(8):
        cy_r = corridor_start_y + i * 8
        if cy_r < WORLD_H - MARGIN - 3:
            corridor_rows.append(cy_r)

    # How many corridors to activate based on pop
    n_corridors_active = min(len(corridor_rows), 1 + pop // 30)

    # Horizontal extent grows with pop
    extent = min(50, 10 + pop // 8)
    x_left = max(MARGIN + 1, plant_cx - extent)
    x_right = min(WORLD_W - MARGIN - 2, plant_cx + extent)

    # ---- Phase 1: Wire vertical spine from plant down to active corridors ----
    if len(corridor_rows) > 0:
        last_cor = corridor_rows[min(len(corridor_rows) - 1, n_corridors_active)]
        for wy in range(plant_cy, last_cor + 1):
            if in_bounds(spine_x, wy) and not power_src[wy, spine_x]:
                adj = any(
                    0 <= wy + dr < WORLD_H and 0 <= spine_x + dc < WORLD_W
                    and power_src[wy + dr, spine_x + dc]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if adj and emp[wy, spine_x]:
                    return (Tool.WIRE, spine_x, wy)

    # ---- Phase 2: Horizontal roads at each active corridor ----
    for ci in range(n_corridors_active):
        cor_y = corridor_rows[ci]
        if not np.any(roa[cor_y, x_left:x_right + 1]):
            if in_bounds(spine_x, cor_y) and emp[cor_y, spine_x]:
                return (Tool.ROAD, spine_x, cor_y)
        road_front = dilate(roa) & emp & margin_mask
        for rx in list(range(spine_x, x_left - 1, -1)) + list(range(spine_x + 1, x_right + 1)):
            if in_bounds(rx, cor_y) and road_front[cor_y, rx]:
                return (Tool.ROAD, rx, cor_y)

    # ---- Phase 3: Vertical road connectors between corridors every 8 cols ----
    for ci in range(max(0, n_corridors_active - 1)):
        cor_y1 = corridor_rows[ci]
        cor_y2 = corridor_rows[ci + 1]
        for rx in range(x_left, x_right + 1, 8):
            if not in_bounds(rx, cor_y1) or not in_bounds(rx, cor_y2):
                continue
            for ry in range(cor_y1 + 1, cor_y2):
                if in_bounds(rx, ry) and emp[ry, rx]:
                    adj_road = any(
                        0 <= ry + dr < WORLD_H and 0 <= rx + dc < WORLD_W
                        and roa[ry + dr, rx + dc]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if adj_road:
                        return (Tool.ROAD, rx, ry)

    # ---- Phase 4: Place zones ----
    def place_zone():
        near_road = dilate_n(roa, 2)
        near_power = power_cov

        candidates = emp & near_road & near_power & margin_mask
        if not np.any(candidates):
            candidates = emp & near_road & margin_mask
        if not np.any(candidates):
            return None

        ys_c, xs_c = np.where(candidates)
        n_res = int(res.sum())
        n_com = int(com.sum())
        n_ind = int(ind.sum())
        total = max(1, n_res + n_com + n_ind)
        res_frac = n_res / total
        com_frac = n_com / total
        ind_frac = n_ind / total

        # Target ~55% res, 25% com, 20% ind
        if pop < 60:
            tool = Tool.RESIDENTIAL
        elif ind_frac < 0.20:
            tool = Tool.INDUSTRIAL
        elif com_frac < 0.25:
            tool = Tool.COMMERCIAL
        elif res_frac < 0.55:
            tool = Tool.RESIDENTIAL
        elif com_frac < 0.30:
            tool = Tool.COMMERCIAL
        else:
            tool = Tool.RESIDENTIAL

        near_zones = dilate(zones_any)
        ref_ys = corridor_rows[:n_corridors_active] if corridor_rows else [corridor_start_y]

        # Score: distance to nearest corridor + small x-distance to spine
        dists = np.array([
            min(abs(int(ys_c[i]) - ry) for ry in ref_ys) + abs(int(xs_c[i]) - spine_x) * 0.1
            for i in range(len(xs_c))
        ])
        zone_bonus = np.where(near_zones[ys_c, xs_c], -2, 4)
        power_pen = np.where(near_power[ys_c, xs_c], 0, 25)
        scores = dists + zone_bonus + power_pen

        sorted_idxs = np.argsort(scores)
        # Prefer clean 3x3 patches
        for idx in sorted_idxs[:400]:
            ty, tx = int(ys_c[idx]), int(xs_c[idx])
            if not in_bounds(tx, ty):
                continue
            y0, y1 = max(0, ty - 1), min(WORLD_H, ty + 2)
            x0, x1 = max(0, tx - 1), min(WORLD_W, tx + 2)
            if y1 - y0 == 3 and x1 - x0 == 3 and np.all(emp[y0:y1, x0:x1]):
                return (tool, tx, ty)
        for idx in sorted_idxs[:60]:
            ty, tx = int(ys_c[idx]), int(xs_c[idx])
            if in_bounds(tx, ty) and emp[ty, tx]:
                return (tool, tx, ty)
        return None

    zone_action = place_zone()
    if zone_action is not None:
        return zone_action

    # ---- Phase 5: Additional power plants as city grows ----
    offsets = [(25, 0), (-25, 0), (0, 25), (0, -15), (25, 25), (-25, 25), (50, 0)]
    thresholds = [40, 120, 300, 500, 800, 1200, 1600]
    for i, (dx, dy) in enumerate(offsets):
        if pop > thresholds[i] and n_plants < i + 2:
            ppx = plant_cx + dx
            ppy = max(MARGIN + 4, min(WORLD_H - MARGIN - 4, plant_cy + dy))
            for ddy in range(-8, 9):
                for ddx in range(-8, 9):
                    nx, ny = ppx + ddx, ppy + ddy
                    if in_bounds(nx, ny) and emp[ny, nx]:
                        return (Tool.COALPOWER, nx, ny)

    # ---- Phase 6: Wire uncovered zones ----
    uncovered = zones_any & ~power_cov
    if np.any(uncovered):
        wire_front = dilate(power_src) & emp & margin_mask
        if np.any(wire_front):
            zy, zx = np.where(uncovered)
            wfy, wfx = np.where(wire_front)
            best_d, best_w = 1e9, None
            for i in range(min(len(wfx), 500)):
                d = int(np.min(np.abs(wfx[i] - zx) + np.abs(wfy[i] - zy)))
                if d < best_d:
                    best_d = d
                    best_w = (int(wfx[i]), int(wfy[i]))
            if best_w and in_bounds(best_w[0], best_w[1]):
                return (Tool.WIRE, best_w[0], best_w[1])

    # ---- Phase 7: Second cluster to the right when pop grows ----
    if pop > 100:
        cluster2_cx = min(plant_cx + 40, WORLD_W - MARGIN - 10)
        cluster2_cy = plant_cy + 3
        cor2_rows = []
        for i in range(4):
            cy_r = cluster2_cy + 5 + i * 8
            if cy_r < WORLD_H - MARGIN - 3:
                cor2_rows.append(cy_r)

        # Wire bridge from spine to cluster2
        bridge_y = plant_cy + 2
        if in_bounds(cluster2_cx, bridge_y):
            for wx in range(spine_x, cluster2_cx + 1):
                if in_bounds(wx, bridge_y) and not power_src[bridge_y, wx]:
                    adj = any(
                        0 <= bridge_y + dr < WORLD_H and 0 <= wx + dc < WORLD_W
                        and power_src[bridge_y + dr, wx + dc]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if adj and emp[bridge_y, wx]:
                        return (Tool.WIRE, wx, bridge_y)

        # Vertical spine for cluster2
        if np.any(power_src[bridge_y, cluster2_cx - 2:cluster2_cx + 3]):
            end_y = cor2_rows[-1] if cor2_rows else bridge_y + 20
            for wy in range(bridge_y, end_y + 1):
                if in_bounds(cluster2_cx, wy) and not power_src[wy, cluster2_cx]:
                    adj = any(
                        0 <= wy + dr < WORLD_H and 0 <= cluster2_cx + dc < WORLD_W
                        and power_src[wy + dr, cluster2_cx + dc]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if adj and emp[wy, cluster2_cx]:
                        return (Tool.WIRE, cluster2_cx, wy)

        # Roads for cluster2
        x2_left = max(MARGIN + 1, cluster2_cx - 20)
        x2_right = min(WORLD_W - MARGIN - 2, cluster2_cx + 20)
        n_cor2_active = min(len(cor2_rows), 1 + pop // 100)
        for cor_y in cor2_rows[:n_cor2_active]:
            if not np.any(roa[cor_y, x2_left:x2_right + 1]):
                if in_bounds(cluster2_cx, cor_y) and emp[cor_y, cluster2_cx]:
                    return (Tool.ROAD, cluster2_cx, cor_y)
            road_front2 = dilate(roa) & emp & margin_mask
            for rx in list(range(cluster2_cx, x2_left - 1, -1)) + list(range(cluster2_cx + 1, x2_right + 1)):
                if in_bounds(rx, cor_y) and road_front2[cor_y, rx]:
                    return (Tool.ROAD, rx, cor_y)

        zone_action2 = place_zone()
        if zone_action2 is not None:
            return zone_action2

    # ---- Phase 8: Extend wire frontier toward uncovered areas ----
    wire_front_ext = dilate(power_src) & emp & margin_mask
    if np.any(wire_front_ext):
        ys_w, xs_w = np.where(wire_front_ext)
        if np.any(zones_any & ~power_cov):
            zy, zx = np.where(zones_any & ~power_cov)
            scores = np.array([int(np.min(np.abs(xs_w[i] - zx) + np.abs(ys_w[i] - zy))) for i in range(len(xs_w))])
        else:
            target_y = corridor_rows[0] if corridor_rows else plant_cy + 10
            scores = np.abs(ys_w - target_y) + np.abs(xs_w - spine_x)
        idx = int(np.argmin(scores))
        bx, by = int(xs_w[idx]), int(ys_w[idx])
        if in_bounds(bx, by):
            return (Tool.WIRE, bx, by)

    # ---- Phase 9: Extend road network ----
    powered_empty = emp & power_cov & margin_mask
    road_front_ext = dilate(roa) & emp & margin_mask
    if np.any(road_front_ext) and np.any(powered_empty):
        ys_r, xs_r = np.where(road_front_ext)
        pey, pex = np.where(powered_empty)
        scores = np.array([int(np.min(np.abs(xs_r[i] - pex) + np.abs(ys_r[i] - pey))) for i in range(min(len(xs_r), 500))])
        idx = int(np.argmin(scores))
        bx, by = int(xs_r[idx]), int(ys_r[idx])
        if in_bounds(bx, by):
            return (Tool.ROAD, bx, by)

    return None
```

## most trajectory-divergent (max div)
- cityPop=2680 (R=88 C=6 I=3) | stored cityPop=2460 | origin=crossover+cl iter=12705
- reactivity: cf_sensitivity=1.00, traj_divergence=0.90, ind_share=0.23  [reactive-map]
- code: 381 lines, 15801 chars | motifs: reads-map, reads-stats, plants=2c, wire, road
```
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
..............CC............................................
..............CC............................................
.............CIICC....................RRRRR.................
...........CCIIIIIC...................RRRRR.................
...........CCIIIIIC...................RRRRR.................
..........CIIIIIIIIRR...............RRRRRCCRRR..............
........CCIIIIIIIIIIICC............RCCCCRCCRCC..............
........CCIIIII*IIIIICC............RCCCC*RRRCC..............
..........CIIIIIIIIRR...............RRRCCRRCC...............
..........CIIIIIIIIRR...............RRRCCRRCC...............
...........RRIIICCC...................RCCRR.................
.............CIICC.....................CCII.................
.............CCC.......................RRII.................
............................................................
............................................................
............................................................
............................................................
............................................................
............................................................
..............CC.......................CC...................
..............CC.......................CC...................
.............RRRRR....................CCRII.................
...........RRRCCCCC.................CCCCRIIR................
...........RRRCCCCC.................CCCCRRRR................
........RRCCRRR*RRRCC..............CCRRR*RRCCR..............
........RRCCRRR.RRRCC..............CCRCC.RRCCR..............
...........RRRCCCCR.................RRCCRRRRR...............
.............RCCCCR...................RCCRR.................
.............RCCCCR...................RCCRR.................
..............II.......................RR...................
..............II.......................RR...................
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
    wir = wire_mask(tm)
    roa = road_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    plt = plant_mask(tm)
    pop = obs.city_pop

    MARGIN = 3

    def in_bounds(x, y):
        return MARGIN <= x < WORLD_W - MARGIN and MARGIN <= y < WORLD_H - MARGIN

    def dilate(mask):
        d = mask.copy()
        d[1:] |= mask[:-1]; d[:-1] |= mask[1:]
        d[:, 1:] |= mask[:, :-1]; d[:, :-1] |= mask[:, 1:]
        return d

    def dilate_n(mask, n):
        d = mask.copy()
        for _ in range(n):
            d = dilate(d)
        return d

    def adj_to(x, y, mask):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny2, nx2 = y+dr, x+dc
            if 0 <= ny2 < WORLD_H and 0 <= nx2 < WORLD_W and mask[ny2, nx2]:
                return True
        return False

    def can_place3x3(cx, cy):
        if not in_bounds(cx, cy): return False
        y0, y1 = cy-1, cy+2
        x0, x1 = cx-1, cx+2
        if y0 < MARGIN or y1 > WORLD_H-MARGIN or x0 < MARGIN or x1 > WORLD_W-MARGIN:
            return False
        return np.all(emp[y0:y1, x0:x1])

    margin_mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    margin_mask[MARGIN:WORLD_H-MARGIN, MARGIN:WORLD_W-MARGIN] = True

    zones_any = res | com | ind
    power_src = plt | wir

    # BFS-style power coverage: power flows through wires/plants/zones
    def bfs_power():
        covered = power_src.copy()
        traversable = wir | plt | res | com | ind
        for _ in range(80):
            nxt = dilate(covered) & traversable | power_src
            if np.array_equal(nxt, covered): break
            covered = nxt
        return covered

    power_cov = bfs_power()

    # Find unique plants (deduplicated by proximity)
    plant_ys, plant_xs = np.where(plt)
    unique_plants = []
    for px2, py2 in zip(plant_xs.tolist(), plant_ys.tolist()):
        already = any(abs(px2-upx) < 6 and abs(py2-upy) < 6 for upx, upy in unique_plants)
        if not already:
            unique_plants.append((px2, py2))

    # Four clusters in grid layout
    clusters = [
        (30, 30), (80, 30),
        (30, 65), (80, 65),
    ]

    # Activate clusters based on population
    n_active = 1
    if pop >= 200: n_active = 2
    if pop >= 600: n_active = 3
    if pop >= 1500: n_active = 4
    active_clusters = clusters[:n_active]

    ext = min(22, 8 + pop // 30)

    # --- PHASE 0: Place first power plant ---
    if not np.any(plt):
        cx, cy = clusters[0]
        return (Tool.COALPOWER, cx, cy)

    # --- PHASE 1: Ensure each active cluster has a plant nearby ---
    for (cx, cy) in active_clusters:
        has_plant = any(abs(px-cx) < 22 and abs(py-cy) < 22 for px, py in unique_plants)
        if not has_plant:
            for r in range(15):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if abs(dx) != r and abs(dy) != r: continue
                        nx2, ny2 = cx+dx, cy+dy
                        if in_bounds(nx2, ny2) and emp[ny2, nx2]:
                            return (Tool.COALPOWER, nx2, ny2)

    # --- PHASE 2: Grow wire spines from each plant outward ---
    # Wire spine: vertical and horizontal from each plant
    for (ppx, ppy) in unique_plants:
        # Which active cluster is this plant serving?
        best_cl = min(active_clusters, key=lambda c: abs(c[0]-ppx)+abs(c[1]-ppy))
        cx, cy = best_cl

        # Spine toward cluster center vertically
        if ppy != cy:
            step = 1 if cy > ppy else -1
            for wy in range(ppy + step, cy + step, step):
                if not in_bounds(ppx, wy): break
                if power_src[wy, ppx]: continue
                if emp[wy, ppx] and adj_to(ppx, wy, power_src):
                    return (Tool.WIRE, ppx, wy)
                elif not emp[wy, ppx]: break

        # Horizontal spine from plant
        for wx in range(ppx+1, min(ppx+ext, WORLD_W-MARGIN)):
            if not in_bounds(wx, ppy): break
            if power_src[ppy, wx]: continue
            if emp[ppy, wx] and adj_to(wx, ppy, power_src):
                return (Tool.WIRE, wx, ppy)
            elif not emp[ppy, wx]: break
        for wx in range(ppx-1, max(ppx-ext, MARGIN), -1):
            if not in_bounds(wx, ppy): break
            if power_src[ppy, wx]: continue
            if emp[ppy, wx] and adj_to(wx, ppy, power_src):
                return (Tool.WIRE, wx, ppy)
            elif not emp[ppy, wx]: break

        # Vertical spine up/down from plant
        for wy in range(ppy-1, max(ppy-ext, MARGIN), -1):
            if not in_bounds(ppx, wy): break
            if power_src[wy, ppx]: continue
            if emp[wy, ppx] and adj_to(ppx, wy, power_src):
                return (Tool.WIRE, ppx, wy)
            elif not emp[wy, ppx]: break
        for wy in range(ppy+1, min(ppy+ext, WORLD_H-MARGIN)):
            if not in_bounds(ppx, wy): break
            if power_src[wy, ppx]: continue
            if emp[wy, ppx] and adj_to(ppx, wy, power_src):
                return (Tool.WIRE, ppx, wy)
            elif not emp[wy, ppx]: break

    # --- PHASE 3: Wire toward unpowered zones reactively ---
    unpowered_zones = zones_any & ~power_cov
    if np.any(unpowered_zones):
        frontier = dilate(power_src) & emp & margin_mask
        if np.any(frontier):
            zy, zx = np.where(unpowered_zones)
            fy, fx = np.where(frontier)
            step = max(1, len(fx) // 300)
            best_d, best_w = 1e9, None
            for i in range(0, len(fx), step):
                d = int(np.min(np.abs(fx[i] - zx) + np.abs(fy[i] - zy)))
                if d < best_d:
                    best_d = d
                    best_w = (int(fx[i]), int(fy[i]))
            if best_w and in_bounds(best_w[0], best_w[1]):
                return (Tool.WIRE, best_w[0], best_w[1])

    # --- PHASE 4: Build road grid around clusters ---
    road_spacing = 4  # every 4 rows/cols place a road
    if not np.any(roa):
        # Start road near first cluster
        cx, cy = active_clusters[0]
        wf = dilate(power_src) & emp & margin_mask
        if np.any(wf):
            fy, fx = np.where(wf)
            dists = np.abs(fx - cx) + np.abs(fy - cy)
            idx = int(np.argmin(dists))
            return (Tool.ROAD, int(fx[idx]), int(fy[idx]))

    for (ppx, ppy) in unique_plants:
        best_cl = min(active_clusters, key=lambda c: abs(c[0]-ppx)+abs(c[1]-ppy))
        cx, cy = best_cl

        # Horizontal roads at regular intervals around cluster
        for offset in range(0, ext+1, road_spacing):
            for road_y in ([cy] if offset == 0 else [cy+offset, cy-offset]):
                if not in_bounds(cx, road_y): continue
                # Grow road left from cx
                for rx in range(cx-1, max(cx-ext-2, MARGIN), -1):
                    if not in_bounds(rx, road_y): break
                    if roa[road_y, rx]: continue
                    if emp[road_y, rx] and adj_to(rx, road_y, roa):
                        return (Tool.ROAD, rx, road_y)
                    elif not emp[road_y, rx]: break
                # Grow road right from cx
                for rx in range(cx+1, min(cx+ext+2, WORLD_W-MARGIN)):
                    if not in_bounds(rx, road_y): break
                    if roa[road_y, rx]: continue
                    if emp[road_y, rx] and adj_to(rx, road_y, roa):
                        return (Tool.ROAD, rx, road_y)
                    elif not emp[road_y, rx]: break

        # Vertical roads at regular intervals
        for offset in range(road_spacing, ext+1, road_spacing):
            for road_x in [cx+offset, cx-offset]:
                if not in_bounds(road_x, cy): continue
                y_top = max(MARGIN, cy-ext)
                y_bot = min(WORLD_H-MARGIN, cy+ext)
                for ry in range(cy, y_bot+1):
                    if not in_bounds(road_x, ry): break
                    if roa[ry, road_x]: continue
                    if emp[ry, road_x] and adj_to(road_x, ry, roa):
                        return (Tool.ROAD, road_x, ry)
                    elif not emp[ry, road_x]: break
                for ry in range(cy-1, y_top-1, -1):
                    if not in_bounds(road_x, ry): break
                    if roa[ry, road_x]: continue
                    if emp[ry, road_x] and adj_to(road_x, ry, roa):
                        return (Tool.ROAD, road_x, ry)
                    elif not emp[ry, road_x]: break

    # Connect clusters with roads
    if n_active >= 2:
        # Horizontal connections between left and right clusters at same row
        for i in range(0, n_active-1, 2):
            if i+1 < n_active:
                cl_l, cl_r = active_clusters[i], active_clusters[i+1]
                conn_y = cl_l[1]
                rrf = dilate(roa) & emp & margin_mask
                for rx in range(cl_l[0]+ext//2, cl_r[0]-ext//2+1):
                    if not in_bounds(rx, conn_y): continue
                    if roa[conn_y, rx]: continue
                    if rrf[conn_y, rx] and emp[conn_y, rx]:
                        return (Tool.ROAD, rx, conn_y)
        # Vertical connections between top and bottom clusters
        if n_active >= 3:
            cl_t, cl_b = active_clusters[0], active_clusters[2]
            conn_x = cl_t[0]
            rrf = dilate(roa) & emp & margin_mask
            for ry in range(cl_t[1]+ext//2, cl_b[1]-ext//2+1):
                if not in_bounds(conn_x, ry): continue
                if roa[ry, conn_x]: continue
                if rrf[ry, conn_x] and emp[ry, conn_x]:
                    return (Tool.ROAD, conn_x, ry)
        if n_active >= 4:
            cl_t, cl_b = active_clusters[1], active_clusters[3]
            conn_x = cl_t[0]
            rrf = dilate(roa) & emp & margin_mask
            for ry in range(cl_t[1]+ext//2, cl_b[1]-ext//2+1):
                if not in_bounds(conn_x, ry): continue
                if roa[ry, conn_x]: continue
                if rrf[ry, conn_x] and emp[ry, conn_x]:
                    return (Tool.ROAD, conn_x, ry)

    # --- PHASE 5: Wire branches along road rows ---
    for (ppx, ppy) in unique_plants:
        best_cl = min(active_clusters, key=lambda c: abs(c[0]-ppx)+abs(c[1]-ppy))
        cx, cy = best_cl
        for offset in range(0, ext+1, road_spacing):
            for branch_y in ([cy] if offset == 0 else [cy+offset, cy-offset]):
                if not (MARGIN <= branch_y < WORLD_H-MARGIN): continue
                if not np.any(power_src[branch_y, :]):
                    continue
                for wx in range(cx-1, max(cx-ext-2, MARGIN), -1):
                    if not in_bounds(wx, branch_y): break
                    if power_src[branch_y, wx]: break
                    if emp[branch_y, wx] and adj_to(wx, branch_y, power_src):
                        return (Tool.WIRE, wx, branch_y)
                    elif not emp[branch_y, wx]: break
                for wx in range(cx+1, min(cx+ext+2, WORLD_W-MARGIN)):
                    if not in_bounds(wx, branch_y): break
                    if power_src[branch_y, wx]: break
                    if emp[branch_y, wx] and adj_to(wx, branch_y, power_src):
                        return (Tool.WIRE, wx, branch_y)
                    elif not emp[branch_y, wx]: break

    # General wire frontier toward unpowered zones
    wire_frontier = dilate(power_src) & emp & margin_mask
    if np.any(wire_frontier) and np.any(zones_any):
        zy, zx = np.where(zones_any & ~power_cov) if np.any(zones_any & ~power_cov) else np.where(zones_any)
        fy, fx = np.where(wire_frontier)
        if len(fx) > 0 and len(zx) > 0:
            step = max(1, len(fx) // 200)
            best_d, best_w = 1e9, None
            for i in range(0, len(fx), step):
                d = int(np.min(np.abs(fx[i]-zx)+np.abs(fy[i]-zy)))
                if d < best_d:
                    best_d = d
                    best_w = (int(fx[i]), int(fy[i]))
            if best_w:
                return (Tool.WIRE, best_w[0], best_w[1])

    # --- PHASE 6: Choose zone type reactively ---
    n_res = int(res.sum()); n_com = int(com.sum()); n_ind = int(ind.sum())
    total_z = max(1, n_res + n_com + n_ind)
    rf = n_res / total_z; cf = n_com / total_z; inf_ = n_ind / total_z

    if pop < 80:
        tool = Tool.INDUSTRIAL
    elif pop < 300:
        if inf_ < 0.33: tool = Tool.INDUSTRIAL
        elif cf < 0.33: tool = Tool.COMMERCIAL
        else: tool = Tool.RESIDENTIAL
    elif pop < 800:
        if cf < 0.35: tool = Tool.COMMERCIAL
        elif inf_ < 0.28: tool = Tool.INDUSTRIAL
        elif rf < 0.37: tool = Tool.RESIDENTIAL
        else: tool = Tool.COMMERCIAL
    elif pop < 2500:
        if rf < 0.44: tool = Tool.RESIDENTIAL
        elif cf < 0.34: tool = Tool.COMMERCIAL
        elif inf_ < 0.22: tool = Tool.INDUSTRIAL
        else: tool = Tool.RESIDENTIAL
    else:
        if rf < 0.52: tool = Tool.RESIDENTIAL
        elif cf < 0.34: tool = Tool.COMMERCIAL
        elif inf_ < 0.14: tool = Tool.INDUSTRIAL
        else: tool = Tool.RESIDENTIAL

    # --- PHASE 7: Place zones near road + power ---
    power_cov = bfs_power()
    near_road = dilate_n(roa, 2)

    candidates = emp & near_road & power_cov & margin_mask
    if not np.any(candidates):
        candidates = emp & near_road & margin_mask
    if not np.any(candidates):
        candidates = emp & power_cov & margin_mask
    if not np.any(candidates):
        candidates = emp & margin_mask

    if np.any(candidates):
        ys, xs = np.where(candidates)
        near_zones = dilate(zones_any)

        dist_to_plants = np.full(len(xs), 1e9)
        for (ppx, ppy) in unique_plants:
            d = np.abs(xs - ppx).astype(float) + np.abs(ys - ppy).astype(float)
            dist_to_plants = np.minimum(dist_to_plants, d)

        dist_to_active = np.full(len(xs), 1e9)
        for (cx, cy) in active_clusters:
            d = np.abs(xs - cx).astype(float) + np.abs(ys - cy).astype(float)
            dist_to_active = np.minimum(dist_to_active, d)

        power_pen = np.where(power_cov[ys, xs], 0.0, 25.0)
        road_pen = np.where(near_road[ys, xs], 0.0, 10.0)
        zone_adj = np.where(near_zones[ys, xs], -3.0, 3.0)
        scores = dist_to_plants * 0.25 + dist_to_active * 0.45 + power_pen + road_pen + zone_adj

        sorted_idxs = np.argsort(scores)
        for idx in sorted_idxs[:500]:
            ty, tx = int(ys[idx]), int(xs[idx])
            if can_place3x3(tx, ty):
                return (tool, tx, ty)
        for idx in sorted_idxs[:50]:
            ty, tx = int(ys[idx]), int(xs[idx])
            if in_bounds(tx, ty) and emp[ty, tx]:
                return (tool, tx, ty)

    # --- PHASE 8: Extend roads outward toward active clusters ---
    road_front = dilate(roa) & emp & margin_mask
    if np.any(road_front):
        ry2, rx2 = np.where(road_front)
        dist_to_active = np.full(len(rx2), 1e9)
        for (cx, cy) in active_clusters:
            d = (rx2 - cx)**2 + (ry2 - cy)**2
            dist_to_active = np.minimum(dist_to_active, d.astype(float))
        idx = int(np.argmin(dist_to_active))
        x, y = int(rx2[idx]), int(ry2[idx])
        if in_bounds(x, y):
            return (Tool.ROAD, x, y)

    # --- PHASE 9: General wire extension toward active clusters ---
    wf = dilate(power_src) & emp & margin_mask
    if np.any(wf):
        fy, fx = np.where(wf)
        dist_to_active = np.full(len(fx), 1e9)
        for (cx, cy) in active_clusters:
            d = np.abs(fx - cx).astype(float) + np.abs(fy - cy).astype(float)
            dist_to_active = np.minimum(dist_to_active, d)
        idx = int(np.argmin(dist_to_active))
        return (Tool.WIRE, int(fx[idx]), int(fy[idx]))

    return None
```

## best open-loop
- cityPop=2200 (R=92 C=0 I=0) | stored cityPop=2224 | origin=crossover iter=797
- reactivity: cf_sensitivity=0.00, traj_divergence=0.00, ind_share=0.14  [reactive-map]
- code: 301 lines, 11053 chars | motifs: precomputed-plan, reads-map, plants=5c, wire, road
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
...............=............................................
..........RRRRR=............................................
..........RRRRR=............................................
..........RRRRR=CC..........................................
..........RRRRR=CCII........................................
..........RRRRR=CCII........................................
..........RRRRR=CCII........................................
..........RRRRR=CCII........................................
.........*.RRRR=CCII...............................*........
...........RRRR=CCII........................................
..........RRRRR=CCII........................................
..........RRRRR=CCII........................................
..........RRRRR=CC..........................................
..........RRRRR=CC..........................................
..........RRRRR=............................................
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
```
```python
def act(obs, state):
    # Strategy: Pre-planned layout with two coal power plants, vertical road+wire spines,
    # and densely packed RCI zones. Then reactive expansion.
    
    def make_plan():
        actions = []
        cy = 50  # center y
        
        # === LEFT NEIGHBORHOOD ===
        # Power plant at left
        pp1x, pp1y = 18, cy
        actions.append((Tool.COALPOWER, pp1x, pp1y))
        
        road_x_l = 30
        wire_x_l = 31
        
        # Horizontal wire from plant to wire column
        for wx in range(pp1x + 2, wire_x_l + 1):
            actions.append((Tool.WIRE, wx, pp1y))
        
        # Vertical road (left neighborhood)
        for dy in range(-15, 16):
            actions.append((Tool.ROAD, road_x_l, cy + dy))
        
        # Vertical wire column
        for dy in range(-15, 16):
            actions.append((Tool.WIRE, wire_x_l, cy + dy))
        
        # Zones left of road (residential blocks, spaced every 3)
        for dy in range(-12, 13, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                # Two columns of zones left of road
                if road_x_l - 3 >= 3:
                    actions.append((Tool.RESIDENTIAL, road_x_l - 3, ny))
                if road_x_l - 6 >= 3:
                    actions.append((Tool.RESIDENTIAL, road_x_l - 6, ny))
                if road_x_l - 9 >= 3:
                    actions.append((Tool.RESIDENTIAL, road_x_l - 9, ny))
        
        # Commercial right of wire column
        for dy in range(-9, 10, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if wire_x_l + 3 <= 116:
                    actions.append((Tool.COMMERCIAL, wire_x_l + 3, ny))
        
        # Industrial further right
        for dy in range(-6, 7, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if wire_x_l + 6 <= 116:
                    actions.append((Tool.INDUSTRIAL, wire_x_l + 6, ny))
        
        # === RIGHT NEIGHBORHOOD ===
        pp2x, pp2y = 102, cy
        actions.append((Tool.COALPOWER, pp2x, pp2y))
        
        road_x_r = 90
        wire_x_r = 89
        
        # Horizontal wire from right plant to wire column
        for wx in range(wire_x_r, pp2x - 1):
            actions.append((Tool.WIRE, wx, pp2y))
        
        # Vertical road (right neighborhood)
        for dy in range(-15, 16):
            actions.append((Tool.ROAD, road_x_r, cy + dy))
        
        # Vertical wire column
        for dy in range(-15, 16):
            actions.append((Tool.WIRE, wire_x_r, cy + dy))
        
        # Zones right of road
        for dy in range(-12, 13, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if road_x_r + 3 <= 116:
                    actions.append((Tool.RESIDENTIAL, road_x_r + 3, ny))
                if road_x_r + 6 <= 116:
                    actions.append((Tool.RESIDENTIAL, road_x_r + 6, ny))
                if road_x_r + 9 <= 116:
                    actions.append((Tool.RESIDENTIAL, road_x_r + 9, ny))
        
        # Commercial left of right wire
        for dy in range(-9, 10, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if wire_x_r - 3 >= 3:
                    actions.append((Tool.COMMERCIAL, wire_x_r - 3, ny))
        
        # Industrial further left of right wire
        for dy in range(-6, 7, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if wire_x_r - 6 >= 3:
                    actions.append((Tool.INDUSTRIAL, wire_x_r - 6, ny))
        
        # === CENTER NEIGHBORHOOD ===
        # Center power plant
        cpx, cpy = 60, cy - 20
        actions.append((Tool.COALPOWER, cpx, cpy))
        
        # Wire from center plant down to center road area
        for wy in range(cpy + 2, cy - 13):
            actions.append((Tool.WIRE, cpx, wy))
        
        # Central vertical road
        cx = 60
        for dy in range(-15, 16):
            actions.append((Tool.ROAD, cx, cy + dy))
        
        # Wire beside central road
        for dy in range(-15, 16):
            actions.append((Tool.WIRE, cx + 1, cy + dy))
        
        # Horizontal connecting roads
        for hx in range(road_x_l, road_x_r + 1):
            actions.append((Tool.ROAD, hx, cy - 15))
            actions.append((Tool.ROAD, hx, cy + 15))
            actions.append((Tool.ROAD, hx, cy))
        
        # Horizontal wires connecting all wire columns at plant levels
        for wx in range(wire_x_l, cx):
            actions.append((Tool.WIRE, wx, cpy))
        for wx in range(cx, wire_x_r + 1):
            actions.append((Tool.WIRE, wx, cpy))
        # Connect center plant wire to left and right wire columns
        for wy in range(cpy, cpy + 1):
            actions.append((Tool.WIRE, wire_x_l, wy))
            actions.append((Tool.WIRE, wire_x_r, wy))
        
        # Commercial zones near center
        for dy in range(-9, 10, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if cx - 4 >= 3:
                    actions.append((Tool.COMMERCIAL, cx - 4, ny))
                if cx + 4 <= 116:
                    actions.append((Tool.COMMERCIAL, cx + 4, ny))
        
        # More residential between neighborhoods
        # Between left neighborhood wire and center
        mid_l = (wire_x_l + cx) // 2
        road_mid_l = mid_l
        for dy in range(-12, 13):
            actions.append((Tool.ROAD, road_mid_l, cy + dy))
        # Wire beside it
        for dy in range(-12, 13):
            actions.append((Tool.WIRE, road_mid_l + 1, cy + dy))
        for dy in range(-9, 10, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if road_mid_l - 3 >= 3:
                    actions.append((Tool.RESIDENTIAL, road_mid_l - 3, ny))
                if road_mid_l + 4 <= 116:
                    actions.append((Tool.RESIDENTIAL, road_mid_l + 4, ny))
        
        # Between center and right
        mid_r = (cx + wire_x_r) // 2
        road_mid_r = mid_r
        for dy in range(-12, 13):
            actions.append((Tool.ROAD, road_mid_r, cy + dy))
        for dy in range(-12, 13):
            actions.append((Tool.WIRE, road_mid_r - 1, cy + dy))
        for dy in range(-9, 10, 3):
            ny = cy + dy
            if 3 <= ny <= 96:
                if road_mid_r - 4 >= 3:
                    actions.append((Tool.RESIDENTIAL, road_mid_r - 4, ny))
                if road_mid_r + 3 <= 116:
                    actions.append((Tool.RESIDENTIAL, road_mid_r + 3, ny))
        
        # Additional power plant bottom center
        actions.append((Tool.COALPOWER, 60, cy + 22))
        for wy in range(cy + 16, cy + 22):
            actions.append((Tool.WIRE, 60, wy))
        for wy in range(cy + 22, cy + 24):
            actions.append((Tool.WIRE, 60, wy))
        
        # Extra zones above/below the main band
        for nx in [road_x_l - 3, road_x_l - 6, cx - 4, cx + 4, road_x_r + 3, road_x_r + 6]:
            for dy_off in [-18, 18]:
                ny = cy + dy_off
                if 3 <= nx <= 116 and 3 <= ny <= 96:
                    actions.append((Tool.RESIDENTIAL, nx, ny))
        
        return actions
    
    if "plan" not in state:
        state["plan"] = make_plan()
        state["idx"] = 0
        state["expand_step"] = 0
    
    if state["idx"] < len(state["plan"]):
        a = state["plan"][state["idx"]]
        state["idx"] += 1
        return a
    
    # === Reactive expansion phase ===
    tm = obs.tile_map
    emp = empty_mask(tm)
    rd = road_mask(tm)
    wires = wire_mask(tm)
    plants = plant_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    
    def expand(mask, steps=1):
        m = mask.copy()
        for _ in range(steps):
            tmp = m.copy()
            tmp[1:, :] |= m[:-1, :]
            tmp[:-1, :] |= m[1:, :]
            tmp[:, 1:] |= m[:, :-1]
            tmp[:, :-1] |= m[:, 1:]
            m = tmp
        return m
    
    margin_mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    margin_mask[3:97, 3:117] = True
    
    powered = plants | wires
    near_power = expand(powered, steps=3)
    near_road = expand(rd, steps=2)
    
    res_count = int(res.sum())
    com_count = int(com.sum())
    ind_count = int(ind.sum())
    zone_count = res_count + com_count + ind_count
    total = max(zone_count + 1, 1)
    
    # Balance RCI: target 50% res, 25% com, 25% ind
    if ind_count / total < 0.20:
        tool = Tool.INDUSTRIAL
    elif com_count / total < 0.25:
        tool = Tool.COMMERCIAL
    else:
        tool = Tool.RESIDENTIAL
    
    step = state["expand_step"]
    state["expand_step"] = step + 1
    
    # Best candidate: empty, near road, near power, in margin, mostly empty 3x3
    good = emp & near_road & near_power & margin_mask
    ys, xs = np.where(good)
    
    if len(xs) > 0:
        # Find one with mostly empty 3x3 neighborhood
        best_idx = None
        best_score = -1
        # Sample up to 20 candidates
        indices = [(obs.step + step + i * 7) % len(xs) for i in range(min(20, len(xs)))]
        for i in indices:
            nx, ny = int(xs[i]), int(ys[i])
            if ny-1 >= 0 and ny+2 <= WORLD_H and nx-1 >= 0 and nx+2 <= WORLD_W:
                score = int(np.sum(emp[ny-1:ny+2, nx-1:nx+2]))
                if score > best_score:
                    best_score = score
                    best_idx = i
        if best_idx is not None and best_score >= 6:
            return (tool, int(xs[best_idx]), int(ys[best_idx]))
        k = int((obs.step + step) % len(xs))
        return (tool, int(xs[k]), int(ys[k]))
    
    # Extend wire toward road areas that lack power
    unpowered_road = near_road & ~near_power & emp & margin_mask
    yr, xr = np.where(unpowered_road)
    if len(xr) > 0:
        wire_frontier = expand(powered, steps=1) & emp & margin_mask
        yf, xf = np.where(wire_frontier)
        if len(xf) > 0:
            tx = float(np.mean(xr))
            ty = float(np.mean(yr))
            dists = (xf - tx)**2 + (yf - ty)**2
            best = int(np.argmin(dists))
            return (Tool.WIRE, int(xf[best]), int(yf[best]))
    
    # Add more roads in populated areas
    any_zone = res | com | ind
    near_zone = expand(any_zone, steps=3)
    fill_road = emp & near_zone & margin_mask & ~near_road
    yf2, xf2 = np.where(fill_road)
    if len(xf2) > 0:
        idx = step % len(xf2)
        return (Tool.ROAD, int(xf2[idx]), int(yf2[idx]))
    
    # Add extra power plant if needed
    if zone_count > 40 and int(np.sum(plants)) < 6:
        py_arr, px_arr = np.where(plants)
        if len(px_arr) > 0:
            pcx = int(np.mean(px_arr))
            pcy = int(np.mean(py_arr))
            candidates = [(pcx + 40, pcy), (pcx - 40, pcy),
                          (pcx, pcy + 20), (pcx, pcy - 20)]
            for npx, npy in candidates:
                if 4 <= npx <= 115 and 4 <= npy <= 95 and emp[npy, npx]:
                    return (Tool.COALPOWER, npx, npy)
    
    return None
```

## most industrial
- cityPop=960 (R=0 C=0 I=4) | stored cityPop=928 | origin=mutate+cl iter=14197
- reactivity: cf_sensitivity=0.68, traj_divergence=0.18, ind_share=1.00  [reactive-map]
- code: 353 lines, 12735 chars | motifs: reads-map, reads-stats, plants=3c, wire, road
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
..............................*.............................
............................II..............................
..........................IIIIIIII..........................
..........................IIIIIIII..........................
.........................IIIIIIIIIII........................
.......................IIIIIIIIIIIII........................
.......................IIIIIIIIIIIII........................
......................IIIIIIIIIIIIIIIII.....................
...................*..IIIIIIIIIIIIIIIIII....................
......................IIIIIIIIIIIIIIIIII....................
......................IIIIIIIIIIIIIIIII.....................
......................IIIIIIIIIIIIIIIII.....................
.......................IIIIIIIIIIIII........................
.........................IIIIIIIII..........................
.........................IIIIIIIII*.........................
..........................IIIIIII...........................
............................IIIII...........................
............................IIIII...........................
.............................II.............................
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
    wir = wire_mask(tm)
    roa = road_mask(tm)
    res = res_mask(tm)
    com = com_mask(tm)
    ind = ind_mask(tm)
    plt = plant_mask(tm)

    MARGIN = 3
    pop = obs.city_pop

    def in_bounds(x, y):
        return MARGIN <= x < WORLD_W - MARGIN and MARGIN <= y < WORLD_H - MARGIN

    def dilate(mask):
        d = mask.copy()
        d[1:, :] |= mask[:-1, :]
        d[:-1, :] |= mask[1:, :]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        return d

    def dilate_n(mask, n):
        d = mask.copy()
        for _ in range(n):
            d = dilate(d)
        return d

    margin_mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    margin_mask[MARGIN:WORLD_H-MARGIN, MARGIN:WORLD_W-MARGIN] = True

    zones_any = res | com | ind
    power_src = plt | wir

    # BFS power coverage through connected power tiles and zones
    def bfs_power():
        covered = power_src.copy()
        traversable = wir | plt | res | com | ind
        for _ in range(100):
            nxt = dilate(covered) & traversable
            new = nxt | power_src
            if np.array_equal(new, covered):
                break
            covered = new
        return covered

    power_cov = bfs_power()
    near_road = dilate_n(roa, 3)

    plant_ys, plant_xs = np.where(plt)
    n_plants = len(plant_xs)
    px_all = plant_xs.tolist()
    py_all = plant_ys.tolist()

    # Central cluster - use a slightly different layout than parent
    CX, CY = 60, 50

    # Denser road grid: rows every 3 tiles, cols every 4 tiles
    road_rows = [CY - 9, CY - 6, CY - 3, CY, CY + 3, CY + 6, CY + 9]
    road_cols = [CX - 16, CX - 12, CX - 8, CX - 4, CX, CX + 4, CX + 8, CX + 12, CX + 16]
    spine_x = CX

    # ---- Phase 0: Place first power plant ----
    if not np.any(plt):
        return (Tool.COALPOWER, CX, CY - 18)

    py0 = int(np.mean(plant_ys)) if len(plant_ys) > 0 else CY - 18

    # ---- Phase 1: Reactively add power plants based on zone coverage ----
    n_zones = int(zones_any.sum())
    pz_ratio = obs.powered_zones / max(1, n_zones)

    def try_add_plant(target_x, target_y, threshold, max_plants):
        if pop > threshold and n_plants < max_plants:
            near = any(abs(px - target_x) < 12 and abs(py - target_y) < 12
                       for px, py in zip(px_all, py_all))
            if not near:
                # Place plant at first empty spot near target
                for r2 in range(0, 8):
                    for dy in range(-r2, r2+1):
                        for dx in range(-r2, r2+1):
                            if abs(dx) != r2 and abs(dy) != r2:
                                continue
                            nx, ny = target_x + dx, target_y + dy
                            if in_bounds(nx, ny) and emp[ny, nx]:
                                return (Tool.COALPOWER, nx, ny)
        return None

    r = try_add_plant(CX - 22, CY - 2, 100, 2)
    if r: return r
    r = try_add_plant(CX + 22, CY - 2, 500, 3)
    if r: return r
    r = try_add_plant(CX, CY + 20, 1200, 4)
    if r: return r
    r = try_add_plant(CX - 22, CY + 20, 2000, 5)
    if r: return r
    r = try_add_plant(CX + 22, CY + 20, 2800, 6)
    if r: return r

    # Reactive plant: if power coverage is poor, add plant near unpowered zones
    if pop > 300 and pz_ratio < 0.45 and n_plants < 8:
        unpow = zones_any & ~power_cov
        if np.any(unpow):
            uy, ux = np.where(unpow)
            ucx = int(np.mean(ux))
            ucy = int(np.mean(uy))
            for r2 in range(4, 18, 2):
                for ddx, ddy in [(r2,0),(-r2,0),(0,r2),(0,-r2),(r2,r2),(-r2,r2),(-r2,-r2),(r2,-r2)]:
                    nx2, ny2 = ucx+ddx, ucy+ddy
                    if in_bounds(nx2, ny2) and emp[ny2, nx2]:
                        return (Tool.COALPOWER, nx2, ny2)

    # ---- Phase 2: Wire from plant toward cluster center (spine) ----
    # Wire BFS frontier toward spine_x, CY
    if not power_src[CY, spine_x]:
        frontier = dilate(power_src) & emp & margin_mask
        if np.any(frontier):
            fy, fx = np.where(frontier)
            dists = np.abs(fx - spine_x) + np.abs(fy - CY)
            idx = int(np.argmin(dists))
            wx, wy = int(fx[idx]), int(fy[idx])
            if in_bounds(wx, wy):
                return (Tool.WIRE, wx, wy)

    # Wire vertically along spine from plant to CY
    for wy in range(min(py0, CY), max(py0, CY) + 1):
        if not in_bounds(spine_x, wy): continue
        if power_src[wy, spine_x]: continue
        if not emp[wy, spine_x]: continue
        adj = any(
            0 <= wy+dr < WORLD_H and 0 <= spine_x+dc < WORLD_W
            and power_src[wy+dr, spine_x+dc]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
        )
        if adj:
            return (Tool.WIRE, spine_x, wy)

    # ---- Phase 3: Build road grid reactively ----
    # Seed roads at spine intersections where power exists
    for ry in road_rows:
        if not in_bounds(spine_x, ry): continue
        if roa[ry, spine_x] or not emp[ry, spine_x]: continue
        if power_src[ry, spine_x] or dilate(power_src)[ry, spine_x]:
            return (Tool.ROAD, spine_x, ry)

    # Extend roads horizontally along road rows
    for ry in road_rows:
        for rx in range(spine_x - 1, CX - 19, -1):
            if not in_bounds(rx, ry): break
            if roa[ry, rx]: continue
            if not emp[ry, rx]: continue
            adj_r = any(
                0 <= ry+dr < WORLD_H and 0 <= rx+dc < WORLD_W and roa[ry+dr, rx+dc]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if adj_r:
                return (Tool.ROAD, rx, ry)
        for rx in range(spine_x + 1, CX + 19):
            if not in_bounds(rx, ry): break
            if roa[ry, rx]: continue
            if not emp[ry, rx]: continue
            adj_r = any(
                0 <= ry+dr < WORLD_H and 0 <= rx+dc < WORLD_W and roa[ry+dr, rx+dc]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if adj_r:
                return (Tool.ROAD, rx, ry)

    # Vertical roads connecting rows
    for rx in road_cols:
        for ry in range(road_rows[0], road_rows[-1] + 1):
            if not in_bounds(rx, ry): continue
            if roa[ry, rx]: continue
            if not emp[ry, rx]: continue
            adj_r = any(
                0 <= ry+dr < WORLD_H and 0 <= rx+dc < WORLD_W and roa[ry+dr, rx+dc]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if adj_r:
                return (Tool.ROAD, rx, ry)

    # ---- Phase 4: Wire along road rows from spine ----
    for ry in road_rows:
        if not in_bounds(spine_x, ry): continue
        if not power_src[ry, spine_x]: continue
        for wx in range(spine_x - 1, CX - 19, -1):
            if not in_bounds(wx, ry): break
            if power_src[ry, wx]: continue
            if emp[ry, wx]:
                return (Tool.WIRE, wx, ry)
            else:
                break
        for wx in range(spine_x + 1, CX + 19):
            if not in_bounds(wx, ry): break
            if power_src[ry, wx]: continue
            if emp[ry, wx]:
                return (Tool.WIRE, wx, ry)
            else:
                break

    # ---- Phase 5: Wire vertical columns ----
    for rx in road_cols:
        if rx == spine_x: continue
        for ry in range(road_rows[0], road_rows[-1] + 1):
            if not in_bounds(rx, ry): continue
            if power_src[ry, rx]: continue
            if not emp[ry, rx]: continue
            adj = any(
                0 <= ry+dr < WORLD_H and 0 <= rx+dc < WORLD_W
                and power_src[ry+dr, rx+dc]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if adj:
                return (Tool.WIRE, rx, ry)

    # ---- Phase 6: Wire toward unpowered zones reactively ----
    uncovered = zones_any & ~power_cov
    if np.any(uncovered):
        frontier = dilate(power_src) & emp & margin_mask
        if np.any(frontier):
            zy, zx = np.where(uncovered)
            fy, fx = np.where(frontier)
            best_d, best_w = 1e9, None
            for i in range(min(len(fx), 400)):
                d = int(np.min(np.abs(fx[i] - zx) + np.abs(fy[i] - zy)))
                if d < best_d:
                    best_d = d
                    best_w = (int(fx[i]), int(fy[i]))
            if best_w and in_bounds(best_w[0], best_w[1]):
                return (Tool.WIRE, best_w[0], best_w[1])

    # ---- Phase 7: Choose zone type reactively based on current ratios ----
    pow_res = int(np.sum(res & power_cov))
    pow_com = int(np.sum(com & power_cov))
    pow_ind = int(np.sum(ind & power_cov))
    pow_total = max(1, pow_res + pow_com + pow_ind)
    pr = pow_res / pow_total
    pc = pow_com / pow_total
    pi = pow_ind / pow_total

    # Observe road coverage of zones to bias placement
    road_res = int(np.sum(res & near_road))
    road_ind = int(np.sum(ind & near_road))

    if pop < 80:
        # Bootstrap with industry
        tool = Tool.INDUSTRIAL if pi < 0.45 else Tool.RESIDENTIAL
    elif pop < 300:
        if pi < 0.30:
            tool = Tool.INDUSTRIAL
        elif pr < 0.50:
            tool = Tool.RESIDENTIAL
        else:
            tool = Tool.COMMERCIAL
    elif pop < 800:
        if pr < 0.50:
            tool = Tool.RESIDENTIAL
        elif pi < 0.25:
            tool = Tool.INDUSTRIAL
        elif pc < 0.25:
            tool = Tool.COMMERCIAL
        else:
            tool = Tool.RESIDENTIAL
    elif pop < 1600:
        if pr < 0.55:
            tool = Tool.RESIDENTIAL
        elif pc < 0.25:
            tool = Tool.COMMERCIAL
        elif pi < 0.20:
            tool = Tool.INDUSTRIAL
        else:
            tool = Tool.RESIDENTIAL
    else:
        if pr < 0.60:
            tool = Tool.RESIDENTIAL
        elif pc < 0.28:
            tool = Tool.COMMERCIAL
        elif pi < 0.12:
            tool = Tool.INDUSTRIAL
        else:
            tool = Tool.RESIDENTIAL

    # ---- Phase 8: Place zones near infrastructure ----
    # Priority: powered AND near road, then near road, then near power
    candidates = emp & near_road & power_cov & margin_mask
    if not np.any(candidates):
        candidates = emp & near_road & margin_mask
    if not np.any(candidates):
        candidates = emp & power_cov & margin_mask
    if not np.any(candidates):
        candidates = emp & margin_mask

    if np.any(candidates):
        ys, xs = np.where(candidates)
        near_zones = dilate(zones_any)

        cx_dist = np.abs(xs - CX).astype(float)
        cy_dist = np.abs(ys - CY).astype(float)
        dist = cx_dist + cy_dist

        # Penalty/bonus signals read directly from current map state
        power_pen = np.where(power_cov[ys, xs], 0.0, 20.0)
        road_pen = np.where(near_road[ys, xs], 0.0, 10.0)
        zone_adj = np.where(near_zones[ys, xs], -4.0, 5.0)

        # Extra bonus for being adjacent to road (tighter than near_road dilate)
        near_road2 = dilate_n(roa, 1)
        road_bonus = np.where(near_road2[ys, xs], -6.0, 0.0)

        scores = dist + power_pen + road_pen + zone_adj + road_bonus

        sorted_idxs = np.argsort(scores)

        # Prefer clean 3x3 empty patches
        for idx in sorted_idxs[:1000]:
            ty, tx = int(ys[idx]), int(xs[idx])
            if not in_bounds(tx, ty): continue
            y0, y1 = ty - 1, ty + 2
            x0, x1 = tx - 1, tx + 2
            if y0 < MARGIN or y1 > WORLD_H - MARGIN or x0 < MARGIN or x1 > WORLD_W - MARGIN:
                continue
            if np.all(emp[y0:y1, x0:x1]):
                return (tool, tx, ty)

        # Fallback: best candidate without 3x3 check
        for idx in sorted_idxs[:80]:
            ty, tx = int(ys[idx]), int(xs[idx])
            if in_bounds(tx, ty):
                return (tool, tx, ty)

    # ---- Phase 9: Extend roads toward cluster center reactively ----
    road_front = dilate(roa) & emp & margin_mask
    if np.any(road_front):
        ry_arr, rx_arr = np.where(road_front)
        dists = (rx_arr - CX) ** 2 + (ry_arr - CY) ** 2
        idx = int(np.argmin(dists))
        x, y = int(rx_arr[idx]), int(ry_arr[idx])
        if in_bounds(x, y):
            return (Tool.ROAD, x, y)

    # ---- Phase 10: Extend wires toward center ----
    wire_front = dilate(power_src) & emp & margin_mask
    if np.any(wire_front):
        wy_arr, wx_arr = np.where(wire_front)
        dists = (wx_arr - CX) ** 2 + (wy_arr - CY) ** 2
        idx = int(np.argmin(dists))
        x, y = int(wx_arr[idx]), int(wy_arr[idx])
        if in_bounds(x, y):
            return (Tool.WIRE, x, y)

    return None
```

