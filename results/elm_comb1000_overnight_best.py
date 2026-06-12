# fitness=127742.2 cityPop=127340 measures=(0.9175084175084175, 0.10999999999999999, 0.0) origin=mutate+cl iter=8494
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
    pop = int(obs.city_pop)
    pz = int(obs.powered_zones)
    power_grid = plants | wires

    def inb(x, y):
        return 0 <= x < WORLD_W and 0 <= y < WORLD_H

    def inb_safe(x, y):
        return 2 <= x < WORLD_W - 2 and 2 <= y < WORLD_H - 2

    def near(mask, x, y, d=2):
        return bool(np.any(mask[max(0,y-d):min(WORLD_H,y+d+1),
                                max(0,x-d):min(WORLD_W,x+d+1)]))

    def can_place_3x3(cx, cy):
        if not inb_safe(cx, cy): return False
        y0, y1 = cy-1, cy+2
        x0, x1 = cx-1, cx+2
        if y0 < 0 or y1 > WORLD_H or x0 < 0 or x1 > WORLD_W: return False
        return bool(np.all(emp[y0:y1, x0:x1]))

    # Layout constants
    X0 = 8
    Y0 = 8
    W = 15       # zone columns per band
    NBANDS = 11
    road_x0 = X0 + 2
    road_x1 = X0 + 3 * W + 2
    H = NBANDS * 7

    # --- Phase 0: Power plant ---
    if not np.any(plants):
        return (Tool.NUCLEARPOWER, X0 - 4, Y0)

    # --- Phase 1: Stadium early ---
    if not state.get("stadium"):
        state["stadium"] = True
        return (Tool.STADIUM, X0 - 4, Y0 + H - 3)

    # --- Phase: Second nuclear when needed ---
    if not state.get("nuke2") and (pop > 8000 or step > 350):
        state["nuke2"] = True
        nx2 = road_x1 + 4
        ny2 = Y0 + H // 2
        if inb(nx2, ny2):
            return (Tool.NUCLEARPOWER, nx2, ny2)

    # --- Phase: Third nuclear ---
    if not state.get("nuke3") and (pop > 35000 or step > 650):
        state["nuke3"] = True
        if inb(X0 - 4, Y0 + 15):
            return (Tool.NUCLEARPOWER, X0 - 4, Y0 + 15)

    # --- Airport late game ---
    if not state.get("airport") and (pop > 55000 or step > 800):
        state["airport"] = True
        ax, ay = road_x1 + 4, Y0 + 5
        if inb(ax + 2, ay + 2):
            return (Tool.AIRPORT, ax + 2, ay + 2)

    # --- Second stadium ---
    if not state.get("stadium2") and pop > 25000:
        state["stadium2"] = True
        return (Tool.STADIUM, X0 - 4, Y0 + 5)

    # --- Seaport ---
    if not state.get("seaport") and pop > 4000:
        state["seaport"] = True
        sx, sy = road_x1 + 4, Y0 + H - 5
        if inb(sx + 1, sy + 1):
            return (Tool.SEAPORT, sx + 1, sy + 1)

    # --- Fire station ---
    if not state.get("fire1") and pop > 200:
        state["fire1"] = True
        fx, fy = X0 - 4, Y0 + 8
        if can_place_3x3(fx, fy):
            return (Tool.FIRESTATION, fx, fy)

    # --- Police station early ---
    if not state.get("police_early") and pop > 500:
        state["police_early"] = True
        fx, fy = road_x1 + 4, Y0 + H // 3
        if can_place_3x3(fx, fy):
            return (Tool.POLICESTATION, fx, fy)

    # --- Reactive extra fire/police ---
    if pop > 5000 and step % 80 == 7 and not state.get(f"fire_extra_{step//80}"):
        state[f"fire_extra_{step//80}"] = True
        fx2, fy2 = X0 - 4, Y0 + 25
        if can_place_3x3(fx2, fy2):
            return (Tool.FIRESTATION, fx2, fy2)

    if pop > 10000 and step % 100 == 13 and not state.get(f"police_extra_{step//100}"):
        state[f"police_extra_{step//100}"] = True
        px2, py2 = road_x1 + 4, Y0 + H * 2 // 3
        if can_place_3x3(px2, py2):
            return (Tool.POLICESTATION, px2, py2)

    # ---- CLOSED-LOOP CORE: scan what exists and extend ----

    # Step A: Ensure top access road exists (reactive check)
    for x in range(road_x0, road_x1):
        if inb(x, Y0 - 1) and emp[Y0 - 1, x]:
            return (Tool.ROAD, x, Y0 - 1)

    # Step B: Spine column - place residential along left spine for power backbone
    for y in range(Y0, Y0 + H, 3):
        if inb(X0, y) and emp[y, X0]:
            return (Tool.RESIDENTIAL, X0, y)

    # Step C: Scan each band and fill roads+zones reactively
    # Determine zone mix from current observed state
    total_res = int(np.sum(res))
    total_com = int(np.sum(com))
    total_ind = int(np.sum(ind))
    total_zones = max(1, total_res + total_com + total_ind)
    r_frac = total_res / total_zones
    c_frac = total_com / total_zones

    if pop < 500:
        target_r, target_c = 0.55, 0.30
    elif pop < 2000:
        target_r, target_c = 0.50, 0.33
    elif pop < 10000:
        target_r, target_c = 0.48, 0.35
    else:
        target_r, target_c = 0.45, 0.38

    def pick_zone_type(k):
        # Use observed fractions to decide
        dr = target_r - r_frac
        dc = target_c - c_frac
        # Force police every 12
        if k > 0 and k % 12 == 0:
            return Tool.POLICESTATION
        # Every 6th is commercial if needed
        if k % 6 == 5:
            return Tool.COMMERCIAL
        # Otherwise use deficit
        if dr >= dc and dr > 0:
            return Tool.RESIDENTIAL
        elif dc > 0:
            return Tool.COMMERCIAL
        return Tool.RESIDENTIAL

    for b in range(NBANDS):
        yb = Y0 + b * 7
        ry = yb + 6
        # Road below band - reactive: only place if missing
        for x in range(road_x0, road_x1):
            if inb(x, ry) and emp[ry, x]:
                return (Tool.ROAD, x, ry)
        # Two zone rows per band
        for r in range(2):
            zy = yb + 1 + 3 * r
            for c in range(W):
                zx = X0 + 3 + c * 3
                if zx > 116 or not inb(zx, zy):
                    continue
                if not emp[zy, zx]:
                    continue
                # Closed-loop check: only place if road or power is nearby
                road_near = near(roads, zx, zy, 4)
                power_near = near(power_grid, zx, zy, 8)
                if road_near or power_near or b == 0:
                    k = b * (2 * W) + r * W + c
                    tool = pick_zone_type(k)
                    return (tool, zx, zy)

    # Step D: Reactive gap-filling using current map state
    # Find zones without road access and build roads toward them
    if step < obs.n_steps - 50:
        # Find unpowered zones and connect wires
        zone_ys, zone_xs = np.where(res | com | ind)
        if len(zone_xs) > 0 and np.any(power_grid):
            pw_ys, pw_xs = np.where(power_grid)
            # Find zones far from power
            for i in range(0, len(zone_xs), max(1, len(zone_xs)//30)):
                zx, zy = int(zone_xs[i]), int(zone_ys[i])
                if not near(power_grid, zx, zy, 5):
                    # Find nearest power point
                    dists = np.abs(pw_xs - zx) + np.abs(pw_ys - zy)
                    idx = int(np.argmin(dists))
                    px, py = int(pw_xs[idx]), int(pw_ys[idx])
                    # Step wire toward zone
                    wx = px + (1 if zx > px else -1 if zx < px else 0)
                    wy = py + (1 if zy > py else -1 if zy < py else 0)
                    if inb(wx, wy) and emp[wy, wx]:
                        return (Tool.WIRE, wx, wy)

        # Find zones without road access - build road nearby
        road_ys_arr, road_xs_arr = np.where(roads)
        if len(road_xs_arr) > 0:
            # Scan for empty tiles near roads and power, fill with zones
            step_r = max(1, len(road_xs_arr) // 80)
            candidates = []
            checked = set()
            for i in range(0, len(road_xs_arr), step_r):
                rx2, ry2 = int(road_xs_arr[i]), int(road_ys_arr[i])
                for dz in range(-4, 5):
                    for dzy in range(-4, 5):
                        zx, zy = rx2 + dz, ry2 + dzy
                        if (zx, zy) in checked: continue
                        checked.add((zx, zy))
                        if zx < 2 or zx >= WORLD_W-2 or zy < 2 or zy >= WORLD_H-2:
                            continue
                        if not emp[zy, zx]: continue
                        y0b, y1b = max(0,zy-1), min(WORLD_H,zy+2)
                        x0b, x1b = max(0,zx-1), min(WORLD_W,zx+2)
                        region = emp[y0b:y1b, x0b:x1b]
                        if region.shape[0] < 3 or region.shape[1] < 3: continue
                        if int(np.sum(region)) < 7: continue
                        road_near2 = near(roads, zx, zy, 2)
                        power_near2 = near(power_grid, zx, zy, 6)
                        if road_near2 and power_near2:
                            dist = abs(zx - WORLD_W//2) + abs(zy - WORLD_H//2)
                            candidates.append((dist, zx, zy))
            if candidates:
                candidates.sort()
                _, zx, zy = candidates[0]
                # Pick zone type reactively
                dr = target_r - r_frac
                dc = target_c - c_frac
                if dr >= dc:
                    desired = Tool.RESIDENTIAL
                else:
                    desired = Tool.COMMERCIAL
                return (desired, zx, zy)

    return None