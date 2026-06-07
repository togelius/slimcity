# fitness=4289.4 cityPop=4280 measures=(0.8757575757575756, 0.3833333333333333, 0.15141467727674626) origin=crossover+cl
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
