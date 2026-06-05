"""The closed-loop champion (cityPop ~2480 replay) from the elm_clind_react
run, used to seed the ambition/init/kill-open-loop follow-up run. Genuinely
closed-loop: act() reads obs.tile_map every step and grows the city reactively.
"""

SEED_CLIND_CHAMP = '''
def act(obs, state):
    """
    Hybrid policy combining best ideas from both parents.
    - Compact cluster with vertical wire spine + flanking roads
    - Balanced zone placement (R/C/I)
    - Second power plant for growth
    - Services at appropriate thresholds
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

    # Fixed anchor: place first coal plant at (58, 50)
    if not np.any(pm):
        return (Tool.COALPOWER, 58, 50)

    plant_ys, plant_xs = np.where(pm)
    # Primary plant center
    px = int(plant_xs[0]) if len(plant_xs) > 0 else 58
    py = int(plant_ys[0]) if len(plant_ys) > 0 else 50

    res_count = int(np.sum(resm))
    com_count = int(np.sum(comm))
    ind_count = int(np.sum(indm))
    total_zones = res_count + com_count + ind_count

    def pick_tool():
        t = max(total_zones, 1)
        i_r = ind_count / t
        c_r = com_count / t
        if i_r < 0.22:
            return Tool.INDUSTRIAL
        if c_r < 0.25:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    # Wire spine column: just right of plant
    wire_col = px + 2
    # Road columns flanking the wire spine
    road_col_l = wire_col - 1
    road_col_r = wire_col + 1

    # Determine vertical extent of wire spine based on population
    pop_factor = min(30, obs.city_pop // 15)
    spine_top = max(3, py - 14 - pop_factor)
    spine_bot = min(WORLD_H - 4, py + 14 + pop_factor)

    # --- Build vertical wire spine ---
    wire_in_col = [r for r in range(WORLD_H)
                   if 0 <= wire_col < WORLD_W and bool(wm[r, wire_col])]

    if not wire_in_col:
        # Seed wire adjacent to plant
        for dy in range(-2, 3):
            ny = py + dy
            if in_bounds(wire_col, ny) and (bool(em[ny, wire_col]) or bool(wm[ny, wire_col])) and not bool(pm[ny, wire_col]):
                return (Tool.WIRE, wire_col, ny)
    else:
        min_wr = min(wire_in_col)
        max_wr = max(wire_in_col)
        # Grow downward
        if max_wr < spine_bot:
            nr = max_wr + 1
            if in_bounds(wire_col, nr) and (bool(em[nr, wire_col]) or bool(wm[nr, wire_col])) and not bool(pm[nr, wire_col]):
                return (Tool.WIRE, wire_col, nr)
        # Grow upward
        if min_wr > spine_top:
            nr = min_wr - 1
            if in_bounds(wire_col, nr) and (bool(em[nr, wire_col]) or bool(wm[nr, wire_col])) and not bool(pm[nr, wire_col]):
                return (Tool.WIRE, wire_col, nr)

    # --- Build road columns flanking the wire spine ---
    # For each wire row, ensure road on left and right
    if wire_in_col:
        step_mod3 = obs.step % 3
        rows_to_check = wire_in_col[::2] if step_mod3 < 2 else wire_in_col[1::2]
        for wr in rows_to_check:
            if in_bounds(road_col_l, wr) and bool(em[wr, road_col_l]) and not bool(pm[wr, road_col_l]):
                return (Tool.ROAD, road_col_l, wr)
            if in_bounds(road_col_r, wr) and bool(em[wr, road_col_r]) and not bool(pm[wr, road_col_r]):
                return (Tool.ROAD, road_col_r, wr)

    # --- Second power plant when city grows ---
    if total_zones > 8 or obs.city_pop > 200:
        p2x = px + 20
        p2y = py
        if in_bounds(p2x, p2y):
            nearby_plant = any(
                in_bounds(p2x + dx, p2y + dy) and bool(pm[p2y + dy, p2x + dx])
                for dy in range(-5, 6) for dx in range(-5, 6)
                if in_bounds(p2x + dx, p2y + dy)
            )
            if not nearby_plant and bool(em[p2y, p2x]):
                return (Tool.COALPOWER, p2x, p2y)
            # Wire horizontal bridge from spine to second plant
            if not nearby_plant:
                for wc in range(wire_col + 1, p2x - 1):
                    if in_bounds(wc, py) and bool(em[py, wc]) and not bool(pm[py, wc]):
                        return (Tool.WIRE, wc, py)
            # Second vertical wire spine
            wire_col2 = p2x - 2
            if wire_in_col:
                for wr in wire_in_col:
                    if in_bounds(wire_col2, wr) and bool(em[wr, wire_col2]) and not bool(pm[wr, wire_col2]):
                        return (Tool.WIRE, wire_col2, wr)

    # --- Build proximity masks for zone placement ---
    nbr_road = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_road[r0:r1, c0:c1] |= rm[sr0:sr1, sc0:sc1]

    nbr_wire = np.zeros((WORLD_H, WORLD_W), dtype=bool)
    for dy in range(-7, 8):
        for dx in range(-7, 8):
            sr0 = max(0, -dy); sr1 = min(WORLD_H, WORLD_H - dy)
            sc0 = max(0, -dx); sc1 = min(WORLD_W, WORLD_W - dx)
            r0 = max(0, dy); r1 = min(WORLD_H, WORLD_H + dy)
            c0 = max(0, dx); c1 = min(WORLD_W, WORLD_W + dx)
            nbr_wire[r0:r1, c0:c1] |= wm[sr0:sr1, sc0:sc1]
            nbr_wire[r0:r1, c0:c1] |= pm[sr0:sr1, sc0:sc1]

    zone_tiles = resm | comm | indm

    def zone_clear(zx, zy):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr2, nc2 = zy + dr, zx + dc
                if not in_bounds(nc2, nr2):
                    return False
                if not bool(em[nr2, nc2]) and not bool(zone_tiles[nr2, nc2]):
                    return False
        return True

    # --- Place zones near roads and power ---
    cand_mask = em & nbr_road & nbr_wire
    cand_mask[0:2, :] = False
    cand_mask[WORLD_H-2:, :] = False
    cand_mask[:, 0:2] = False
    cand_mask[:, WORLD_W-2:] = False

    cys, cxs = np.where(cand_mask)
    if len(cxs) > 0:
        # Prioritize candidates near wire column
        dists = np.abs(cxs - wire_col) + np.abs(cys - py) // 3
        order = np.argsort(dists)
        step_off = int(obs.step * 7) % max(len(order), 1)
        checked = 0
        for i in range(len(order)):
            k = order[(step_off + i) % len(order)]
            zx, zy = int(cxs[k]), int(cys[k])
            if zone_clear(zx, zy):
                return (pick_tool(), zx, zy)
            checked += 1
            if checked > 80:
                break

    # --- Extend roads near wire to fill gaps ---
    road_ys2, road_xs2 = np.where(rm)
    if len(road_xs2) > 0:
        step_k = int(obs.step * 11) % max(len(road_xs2), 1)
        for offset in range(min(len(road_xs2), 20)):
            k = (step_k + offset * 7) % len(road_xs2)
            rx2, ry2 = int(road_xs2[k]), int(road_ys2[k])
            for dx2, dy2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx2, ny2 = rx2 + dx2, ry2 + dy2
                if in_bounds(nx2, ny2) and bool(em[ny2, nx2]) and not bool(pm[ny2, nx2]) and bool(nbr_wire[ny2, nx2]):
                    return (Tool.ROAD, nx2, ny2)

    # --- Services when pop is high enough ---
    if obs.city_pop > 150:
        for fpos in [(px - 10, py - 10), (px + 10, py - 10)]:
            fsx, fsy = fpos
            if in_bounds(fsx, fsy) and bool(em[fsy, fsx]):
                return (Tool.FIRESTATION, fsx, fsy)
    if obs.city_pop > 350:
        for ppos in [(px - 10, py + 10), (px + 10, py + 10)]:
            psx, psy = ppos
            if in_bounds(psx, psy) and bool(em[psy, psx]):
                return (Tool.POLICESTATION, psx, psy)

    # --- Expand wire laterally as fallback ---
    wire_ys_arr, wire_xs_arr = np.where(wm)
    if len(wire_xs_arr) > 0:
        step_k2 = int(obs.step * 13) % max(len(wire_xs_arr), 1)
        for offset in range(min(len(wire_xs_arr), 15)):
            k = (step_k2 + offset * 5) % len(wire_xs_arr)
            wx2, wy2 = int(wire_xs_arr[k]), int(wire_ys_arr[k])
            for dx3, dy3 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx3, ny3 = wx2 + dx3, wy2 + dy3
                if in_bounds(nx3, ny3) and bool(em[ny3, nx3]) and not bool(pm[ny3, nx3]):
                    return (Tool.WIRE, nx3, ny3)

    return None
'''
