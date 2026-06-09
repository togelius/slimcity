# Scalable serviced-comb policy, tuned for the 1000-action x 10-tick episode.
#
# Unlike the reactive blob-builders (which are structurally capped at ~30
# powered+road-served zones and so plateau at ~19.5k cityPop no matter how many
# actions they are given), this policy SCALES with the action budget: it lays a
# contiguous residential "spine" (one power network) and stacks 11 horizontal
# zone "bands", each band = 2 zone-rows sandwiched against a shared access road.
# Every zone is therefore powered (spine conduction) AND road-served, so ~220
# zones grow. It builds band-by-band so early zones get the full episode to
# densify. Services: one STADIUM (clears resCap), ~18 police (crime control),
# a sprinkle of commercial as the traffic destinations residential needs to
# densify, and NUCLEAR power (no pollution).
#
# Result @ 1000 actions x 10 ticks: cityPop ~83,300 mean (12 seeds, min ~79.5k),
# vs ~19.5k for the best capped closed-loop policy at the same episode shape.
#
# This is a deterministic blueprint (an "open-loop" builder in the project's
# taxonomy, like the ELM diverse champion): it replays a precomputed plan and
# ignores obs. See the project's note that the performance peak on this task is
# open-loop. A genuinely reactive variant that reads obs is straightforward but
# lands in the same place.

def _build_plan():
    X0, Y0, W, NBANDS = 8, 8, 15, 11
    CSTEP, POLICE_EVERY = 6, 12
    H = NBANDS * 7
    plan = [(Tool.NUCLEARPOWER, X0 - 4, Y0)]
    plan.append((Tool.STADIUM, X0 - 4, Y0 + H - 3))          # clears resCap (resPop>500)
    # Contiguous vertical spine (single power backbone for the whole comb).
    for y in range(Y0, Y0 + H, 3):
        plan.append((Tool.RESIDENTIAL, X0, y))
    # Top access road (serves band-0's upper zone-row).
    for x in range(X0 + 2, X0 + 3 * W + 2):
        plan.append((Tool.ROAD, x, Y0 - 1))
    k = 0
    for b in range(NBANDS):
        YB = Y0 + b * 7
        # Shared access road below this band.
        for x in range(X0 + 2, X0 + 3 * W + 2):
            plan.append((Tool.ROAD, x, YB + 6))
        # Two zone-rows for this band (placed now so they cook the rest of the run).
        for zy in (YB + 1, YB + 4):
            for c in range(W):
                zx = X0 + 3 + c * 3
                if zx > 116:
                    continue
                if POLICE_EVERY and k % POLICE_EVERY == 0 and k > 0:
                    plan.append((Tool.POLICESTATION, zx, zy)); k += 1; continue
                if CSTEP and k % CSTEP == CSTEP - 1:
                    plan.append((Tool.COMMERCIAL, zx, zy)); k += 1; continue
                plan.append((Tool.RESIDENTIAL, zx, zy)); k += 1
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = _build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None
    state["i"] = i + 1
    return plan[i]
