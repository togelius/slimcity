# SlimCity ELM run — elm_clind_long

Quality-Diversity (MAP-Elites) over **Python-code genomes**, with **Claude (Sonnet) as the mutation/crossover operator**. Each genome is a closed-loop `act(obs, state)` tile-placement policy; fitness = cityPop growth (dense-shaped), behavior = (residential share, industrial share); fitness is the mean of 5 independent rolls (the engine eval is ~19% noisy per process).

## Headline
- **Best city: cityPop 14908** (robust mean), fitness 14941, found at iter 9308 via **mutate+cl**, behavior (cf_sensitivity=0.65, traj_divergence=0.00, ind_share=0.00).
- For comparison, the prior CMA-ME state of the art was cityPop 1,680 (layout genome) / 1,120 (action tape). **ELM beats it.**
- Archive: **417 cells filled** (65.2% of 640), QD-score **999798**.
- Iterations completed: **15000**. Genomes that grew a city (pop>0): **7064/14972** evaluated.

## Operator efficiency
- eval=14972  op_error=30  invalid=0  (of 15002 attempts)
- operator failure rate: 0% op-error, 0% invalid.

## Top cities (robust mean over 5 rolls)

| rank | cityPop | fitness | cf_sensitivity | traj_divergence | ind_share | origin | iter |
|---|---|---|---|---|---|---|---|
| 1 | 14908 | 14941 | 0.65 | 0.00 | 0.00 | mutate+cl | 9308 |
| 2 | 11700 | 11738 | 0.85 | 0.08 | 0.02 | mutate+cl | 13778 |
| 3 | 11644 | 11672 | 0.49 | 0.00 | 0.00 | mutate | 12134 |
| 4 | 11136 | 11158 | 0.46 | 0.18 | 0.00 | crossover | 11259 |
| 5 | 10508 | 10540 | 0.66 | 0.32 | 0.01 | crossover+cl | 14843 |
| 6 | 10180 | 10205 | 0.58 | 0.01 | 0.00 | mutate+cl | 12891 |
| 7 | 9760 | 9782 | 0.46 | 0.30 | 0.00 | crossover+cl | 12739 |
| 8 | 9408 | 9425 | 1.00 | 0.00 | 0.00 | crossover | 7932 |
| 9 | 9296 | 9365 | 1.00 | 0.33 | 0.25 | mutate | 11245 |
| 10 | 8800 | 8821 | 0.43 | 0.09 | 0.25 | crossover+cl | 4647 |

## QD-score / coverage progress

| iter | filled | coverage | qd_score | best_pop |
|---|---|---|---|---|
| 0 | 2 | 0.003 | 564 | 336 |
| 1241 | 39 | 0.061 | 38112 | 4796 |
| 2484 | 84 | 0.131 | 113325 | 4796 |
| 3724 | 137 | 0.214 | 194101 | 5528 |
| 4965 | 193 | 0.302 | 277842 | 8800 |
| 6207 | 243 | 0.380 | 358909 | 9344 |
| 7454 | 287 | 0.448 | 459729 | 11960 |
| 8694 | 313 | 0.489 | 539479 | 11960 |
| 9938 | 345 | 0.539 | 668135 | 14908 |
| 11186 | 371 | 0.580 | 742920 | 14908 |
| 12429 | 388 | 0.606 | 842370 | 14908 |
| 13671 | 399 | 0.623 | 919576 | 14908 |
| 14912 | 417 | 0.652 | 986393 | 14908 |
| 15000 | 417 | 0.652 | 999798 | 14908 |

## Origin of elites
- crossover=81, crossover+cl=82, mutate=95, mutate+cl=159

## Champion policy (source)
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

Artifacts: `results/elm_clind_long.npz` (archive, every cell-winner's full source), `results/elm_clind_long_generations.jsonl` (every genome generated), `results/elm_clind_long_champion.py`.
