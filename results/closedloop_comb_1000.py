# Closed-loop (obs-reactive) scalable serviced-comb policy for 1000 x 10 episodes.
#
# Genuinely closed-loop counterpart of results/scalable_comb_1000.py: every step
# it READS obs.tile_map and places the single next tile the live map is missing,
# in a fixed build order, adapting to auto-bulldoze / grown zones / perturbation.
# Grows the same structure (so it matches the blueprint's ~83k cityPop): one
# contiguous residential spine (single power network) + 11 horizontal zone bands,
# each band = 2 zone-rows against a shared access road, built band-by-band so
# early zones cook the whole episode. Services: stadium (clears resCap), periodic
# police (crime), commercial traffic targets (let residential densify), nuclear.
#
# cityPop ~83k mean @ 1000 actions x 10 ticks, vs ~19.5k for the prior capped
# closed-loop champion at the same episode shape.

X0 = 8
Y0 = 8
W = 15
NBANDS = 11
CSTEP = 6
POLICE_EVERY = 12


def act(obs, state):
    tm = obs.tile_map
    emp = empty_mask(tm)
    road_x0 = X0 + 2
    road_x1 = X0 + 3 * W + 2
    H = NBANDS * 7

    # 1. Power plant (only if none on the map).
    if not np.any(plant_mask(tm)):
        return (Tool.NUCLEARPOWER, X0 - 4, Y0)

    # 2. Stadium, once -- clears resCap (resPop>500 otherwise stalls residential).
    if not state.get("stadium"):
        state["stadium"] = True
        return (Tool.STADIUM, X0 - 4, Y0 + H - 3)

    # 3. Contiguous spine cells (power backbone): first missing one.
    for y in range(Y0, Y0 + H, 3):
        if emp[y, X0]:
            return (Tool.RESIDENTIAL, X0, y)

    # 4. Top access road (band 0's upper row): first missing one.
    for x in range(road_x0, road_x1):
        if emp[Y0 - 1, x]:
            return (Tool.ROAD, x, Y0 - 1)

    # 5. Bands in order (band-by-band -> early zones get full cook time).
    for b in range(NBANDS):
        yb = Y0 + b * 7
        ry = yb + 6
        for x in range(road_x0, road_x1):                 # 5a. shared road below band
            if emp[ry, x]:
                return (Tool.ROAD, x, ry)
        for r in range(2):                                # 5b. two zone-rows
            zy = yb + 1 + 3 * r
            for c in range(W):
                zx = X0 + 3 + c * 3
                if zx > 116:
                    continue
                if emp[zy, zx]:
                    k = b * (2 * W) + r * W + c            # position-based slot index
                    if POLICE_EVERY and k % POLICE_EVERY == 0 and k > 0:
                        return (Tool.POLICESTATION, zx, zy)
                    if CSTEP and k % CSTEP == CSTEP - 1:
                        return (Tool.COMMERCIAL, zx, zy)
                    return (Tool.RESIDENTIAL, zx, zy)

    return None
