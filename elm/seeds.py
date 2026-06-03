"""Hand-written starter genomes for the ELM archive.

Each is a source string defining `act(obs, state)` per the contract in
sandbox.PRIMITIVES_CARD. SEED_PLAN is the recommended starting point: a
readable, budget-aware closed-loop builder that produces a small but real
city (nonzero cityPop), giving both MAP-Elites a non-degenerate elite and the
LLM operator a working template to extend. SEED_RANDOM is the honest
random-loop baseline (fitness ~0) for "can the operator bootstrap growth from
nothing?" experiments.
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

SEEDS = {
    "plan": SEED_PLAN,
    "random": SEED_RANDOM,
}
