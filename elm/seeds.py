"""Hand-written starter genomes for the ELM archive.

Each is a source string defining `act(obs, state)` per the contract in
sandbox.PRIMITIVES_CARD. SEED_PLAN is the recommended single starting point: a
readable, budget-aware closed-loop builder that produces a small but real
city (nonzero cityPop), giving both MAP-Elites a non-degenerate elite and the
LLM operator a working template to extend. SEED_RANDOM is the honest
random-loop baseline (fitness ~0) for "can the operator bootstrap growth from
nothing?" experiments.

Beyond those two, this file defines a DIVERSE starter set (see DIVERSE_SEEDS /
the keys registered in SEEDS) so an ELM run can begin from a population that
already spans behavior space — different zone mixes (R-only, C-only, I-only,
balanced R+C+I, R+I jobs/housing) AND different layouts (a single powered
spine, a road+wire grid, two separated neighborhoods, and two obs-reactive
builders that grow zones next to whatever they have already built/powered).
Under the `res_ind` measures (= res_share, ind_share of zoned tiles) these land
in distinct cells, giving MAP-Elites a broad, non-degenerate seed front and the
LLM operator a varied set of templates to mutate and recombine.

Two genuinely CLOSED-LOOP founders are included: `reactive` (extends the
built-up area off existing residential tiles) and `reactive_wire` (zones next
to live wire tiles). They exist so a reactive lineage is present for the
closed-loop directive / closed-loopness descriptors to evolve from.
"""

# A compact, powered, road-served neighborhood built one tile per step.
# Strategy (all driven off a precomputed plan stashed in `state`):
#   1. one coal plant,
#   2. a vertical WIRE spine next to it (wires conduct power; roads don't),
#   3. a ROAD spine for access,
#   4. RESIDENTIAL zones flanking the spine, each 3x3 zone adjacent to wire+road.
# With ~120 steps this finishes comfortably and leaves zones powered+served,
# so they grow. Lots of obvious knobs for the operator: spacing, zone mix,
# cluster size/position, ratio of R/C/I.
SEED_PLAN = '''
def build_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    plan = []
    # Coal plant just left of the spine so it powers the wire column.
    plan.append((Tool.COALPOWER, cx - 2, cy))
    # Vertical infrastructure spine: wire at column cx, road at column cx+1.
    for dy in range(-12, 13):
        plan.append((Tool.WIRE, cx, cy + dy))
        plan.append((Tool.ROAD, cx + 1, cy + dy))
    # Residential zones flanking the spine every 3 rows. Centers at cx-3 (left)
    # and cx+3 (right) put each 3x3 zone's inner edge against wire/road.
    for dy in range(-12, 13, 3):
        plan.append((Tool.RESIDENTIAL, cx - 3, cy + dy))
        plan.append((Tool.RESIDENTIAL, cx + 3, cy + dy))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None                      # plan finished: place nothing
    state["i"] = i + 1
    return plan[i]                       # (tool, x, y); coords clamped if needed
'''

# Honest random placement loop — the degenerate baseline (expected cityPop=0).
SEED_RANDOM = '''
def act(obs, state):
    if "rng" not in state:
        state["rng"] = np.random.default_rng(0)
    rng = state["rng"]
    tools = [Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL,
             Tool.ROAD, Tool.WIRE, Tool.COALPOWER, Tool.PARK]
    tool = tools[int(rng.integers(len(tools)))]
    x = int(rng.integers(WORLD_W))
    y = int(rng.integers(WORLD_H))
    return (tool, x, y)
'''

# A genuinely CLOSED-LOOP starter: a short open-loop bootstrap to get a powered
# base (so zones can grow at all), then per-step REACTIVE placement that reads
# the CURRENT (growing, seed-dependent) map and extends the built-up area. This
# gives the archive a reactive founder (cf_sensitivity>0, trajectory_divergence>0)
# so a closed-loop lineage exists for the operator to evolve from. ~250 cityPop.
SEED_REACTIVE = '''
def base_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    p = [(Tool.COALPOWER, cx - 2, cy)]
    for dy in range(-10, 11):
        if dy != 0:
            p.append((Tool.WIRE, cx, cy + dy))
        p.append((Tool.ROAD, cx + 1, cy + dy))
    for dy in range(-9, 10, 3):
        p.append((Tool.RESIDENTIAL, cx - 3, cy + dy))
    return p

def act(obs, state):
    # Phase 1: lay a small powered base (bootstrap so zones can grow).
    if "p" not in state:
        state["p"] = base_plan(); state["i"] = 0
    if state["i"] < len(state["p"]):
        a = state["p"][state["i"]]; state["i"] += 1
        return a
    # Phase 2: CLOSED-LOOP. React to the CURRENT map: place a residential zone on
    # an empty tile adjacent to an existing residential tile, extending the city
    # where it has actually grown. The choice depends on the live observation, so
    # actions change as the city develops (and differ across worlds).
    tm = obs.tile_map
    res = res_mask(tm); emp = empty_mask(tm)
    nbr = np.zeros_like(emp)
    nbr[1:, :] |= res[:-1, :]; nbr[:-1, :] |= res[1:, :]
    nbr[:, 1:] |= res[:, :-1]; nbr[:, :-1] |= res[:, 1:]
    cand = emp & nbr
    ys, xs = np.where(cand)
    if len(xs) == 0:
        return None
    k = int(tm.sum()) % len(xs)
    return (Tool.RESIDENTIAL, int(xs[k]), int(ys[k]))
'''


# --------------------------------------------------------------------------
# Diverse spine seeds: the same powered-spine skeleton as SEED_PLAN, but with
# different zone CYCLES flanking the spine. The cycle controls the R/C/I mix and
# therefore which (res_share, ind_share) cell the seed lands in. This is the
# cheapest way to spread the starting front across behavior space while keeping
# every seed a genuinely growing, powered+road-served city.
# --------------------------------------------------------------------------
def _spine_seed(zone_cycle_expr: str) -> str:
    return '''
def build_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    cycle = [%s]
    plan = []
    # Coal plant left of the wire spine so the whole column is powered.
    plan.append((Tool.COALPOWER, cx - 2, cy))
    # Vertical spine: WIRE at column cx (conducts power), ROAD at cx+1 (access).
    for dy in range(-12, 13):
        plan.append((Tool.WIRE, cx, cy + dy))
        plan.append((Tool.ROAD, cx + 1, cy + dy))
    # 3x3 zones flanking the spine every 3 rows; the cycle sets each one's type.
    k = 0
    for dy in range(-12, 13, 3):
        plan.append((cycle[k %% len(cycle)], cx - 3, cy + dy))
        plan.append((cycle[(k + 1) %% len(cycle)], cx + 3, cy + dy))
        k += 1
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None
    state["i"] = i + 1
    return plan[i]
''' % zone_cycle_expr


# Industrial-only spine -> (res_share~0, ind_share~1): the high-ind corner.
SEED_INDUSTRIAL = _spine_seed("Tool.INDUSTRIAL")
# Commercial-only spine -> (res~0, ind~0): the low-low corner (com fills the rest).
SEED_COMMERCIAL = _spine_seed("Tool.COMMERCIAL")
# Balanced R/C/I rotation -> middle of behavior space (~0.33, ~0.33).
SEED_MIXED = _spine_seed("Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL")
# Jobs + housing, no commerce -> the (0.5, 0.5) diagonal.
SEED_RES_IND = _spine_seed("Tool.RESIDENTIAL, Tool.INDUSTRIAL")
# Housing-heavy with a little industry -> high-res, low-ind region.
SEED_RES_HEAVY = _spine_seed(
    "Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.RESIDENTIAL, Tool.INDUSTRIAL")


# --------------------------------------------------------------------------
# A road+wire GRID covering a rectangular block, with zones dropped into the
# grid interiors. Structurally different layout from the spine seeds: power and
# access reach a 2-D area rather than a single column, so it can host a denser,
# wider city. Zones alternate R / C / I across the grid for a mixed profile.
# --------------------------------------------------------------------------
SEED_GRID = '''
def build_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    x0, y0 = cx - 12, cy - 9          # top-left of the block
    plan = []
    # One coal plant, wired into the grid's first column.
    plan.append((Tool.COALPOWER, x0 - 2, y0))
    # Horizontal road lines every 4 rows; vertical WIRE lines every 4 cols.
    for gy in range(0, 19):
        for gx in range(0, 25):
            x, y = x0 + gx, y0 + gy
            if gy % 4 == 0:
                plan.append((Tool.ROAD, x, y))
            elif gx % 4 == 0:
                plan.append((Tool.WIRE, x, y))
    # Drop a 3x3 zone into the centre of each grid cell; rotate R/C/I.
    cyc = [Tool.RESIDENTIAL, Tool.COMMERCIAL, Tool.INDUSTRIAL]
    k = 0
    for gy in range(2, 18, 4):
        for gx in range(2, 24, 4):
            plan.append((cyc[k % 3], x0 + gx, y0 + gy))
            k += 1
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None
    state["i"] = i + 1
    return plan[i]
'''


# --------------------------------------------------------------------------
# Two separated neighborhoods: a RESIDENTIAL cluster on the left and an
# INDUSTRIAL cluster on the right, each with its own plant + wire + road, joined
# by a road. Tests the "functional separation" idea and lands near the R+I
# diagonal but via a very different spatial structure than the spine.
# --------------------------------------------------------------------------
SEED_TWIN = '''
def build_plan():
    cx, cy = WORLD_W // 2, WORLD_H // 2
    plan = []
    def cluster(bx, zone):
        out = [(Tool.COALPOWER, bx - 2, cy)]
        for dy in range(-9, 10):
            out.append((Tool.WIRE, bx, cy + dy))
            out.append((Tool.ROAD, bx + 1, cy + dy))
        for dy in range(-9, 10, 3):
            out.append((zone, bx - 3, cy + dy))
            out.append((zone, bx + 3, cy + dy))
        return out
    plan += cluster(cx - 18, Tool.RESIDENTIAL)   # housing on the left
    plan += cluster(cx + 18, Tool.INDUSTRIAL)    # jobs on the right
    # Connect the two neighborhoods with a road so traffic can flow.
    for x in range(cx - 18, cx + 19):
        plan.append((Tool.ROAD, x, cy))
    return plan

def act(obs, state):
    if "plan" not in state:
        state["plan"] = build_plan()
        state["i"] = 0
    i = state["i"]
    plan = state["plan"]
    if i >= len(plan):
        return None
    state["i"] = i + 1
    return plan[i]
'''


# --------------------------------------------------------------------------
# Obs-REACTIVE builder (second closed-loop founder): lays a power+road spine for
# the first chunk of steps, then for the rest of the episode reads obs.tile_map
# and drops a new RESIDENTIAL zone next to an already-powered WIRE tile, growing
# the city outward from what is already working. Distinct from SEED_REACTIVE,
# which extends off existing residential tiles; this one keys on live power.
# --------------------------------------------------------------------------
SEED_REACTIVE_WIRE = '''
def act(obs, state):
    cx, cy = WORLD_W // 2, WORLD_H // 2
    # Phase 1: build the powered spine (coal plant + wire + road column).
    if "phase" not in state:
        spine = [(Tool.COALPOWER, cx - 2, cy)]
        for dy in range(-14, 15):
            spine.append((Tool.WIRE, cx, cy + dy))
            spine.append((Tool.ROAD, cx + 1, cy + dy))
        state["spine"] = spine
        state["i"] = 0
        state["phase"] = 1
        state["placed"] = set()
    if state["phase"] == 1:
        i = state["i"]
        if i < len(state["spine"]):
            state["i"] = i + 1
            return state["spine"][i]
        state["phase"] = 2
    # Phase 2: react to the map. Find a wire tile and zone the cell to its left
    # if we haven't already, so each new zone is adjacent to live power + road.
    tm = obs.tile_map
    ys, xs = np.where(wire_mask(tm))
    for wy, wx in zip(ys.tolist(), xs.tolist()):
        zx, zy = wx - 3, wy
        key = (zx, zy)
        if key in state["placed"]:
            continue
        if 1 <= zx < WORLD_W - 1 and 1 <= zy < WORLD_H - 1:
            state["placed"].add(key)
            return (Tool.RESIDENTIAL, zx, zy)
    return None
'''


from elm.seed_clind import SEED_CLIND_CHAMP   # closed-loop champ (~2480) from elm_clind_react

SEEDS = {
    "plan": SEED_PLAN,
    "random": SEED_RANDOM,
    "clind_champ": SEED_CLIND_CHAMP,      # reactive champion to seed the follow-up run
    "reactive": SEED_REACTIVE,            # closed-loop founder (residential-extend)
    "industrial": SEED_INDUSTRIAL,
    "commercial": SEED_COMMERCIAL,
    "mixed": SEED_MIXED,
    "res_ind": SEED_RES_IND,
    "res_heavy": SEED_RES_HEAVY,
    "grid": SEED_GRID,
    "twin": SEED_TWIN,
    "reactive_wire": SEED_REACTIVE_WIRE,  # closed-loop founder (wire-adjacent)
}

# The diverse starting front used by long ELM runs (everything except the
# pure-noise `random` baseline, which is kept available for bootstrap studies).
DIVERSE_SEEDS = [
    "plan", "industrial", "commercial", "mixed", "res_ind",
    "res_heavy", "grid", "twin", "reactive", "reactive_wire",
]
