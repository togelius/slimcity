"""Tiny convolutional policy for MicropolisEnv.

Designed for CMA-ME: parameters live in a flat float vector so an evolution
strategy can ask/tell. No autograd; only forward pass in NumPy.

Architecture:
    obs (4, H_obs, W_obs) -> 3x3 conv -> (n_tools, H_obs, W_obs) logits
    argmax over flattened logits -> (tool, y_obs, x_obs)
    place at (x_obs * stride + stride//2, y_obs * stride + stride//2)

Observation: one-hot per category, max-pooled over `stride` x `stride` blocks.
    Channel 0: any residential tile
    Channel 1: any commercial tile
    Channel 2: any industrial tile
    Channel 3: any infrastructure (road/wire/rail/power-plant) tile
"""

from __future__ import annotations

import numpy as np

from slimcity import WORLD_W, WORLD_H, Tool

# --- tile-id ranges, from src/micropolis.h ---
# These are tile IDs in the low 10 bits of getTile(). Ranges are approximate
# but cover the bulk of category occupation; close enough for a coarse obs.
RES_RANGE = (240, 422)   # residential zones and houses (LHTHR..LASTRES)
COM_RANGE = (423, 611)   # commercial zones (COMBASE..LASTCOM)
IND_RANGE = (612, 692)   # industrial zones
ROAD_RANGE = (64, 206)   # roads, rails, wires
PLANT_TILES = {750, 816} # coal plant + nuclear plant tile centers

OBS_STRIDE = 5  # divides both WORLD_H (100) and WORLD_W (120)
OBS_H = WORLD_H // OBS_STRIDE  # 20
OBS_W = WORLD_W // OBS_STRIDE  # 24
N_CHAN = 4
N_TOOLS = 20  # see Tool class
# We use a 3x3 conv with SAME padding from (N_CHAN, OBS_H, OBS_W)
# to (N_TOOLS, OBS_H, OBS_W).
KSIZE = 3
N_PARAMS = N_CHAN * N_TOOLS * KSIZE * KSIZE + N_TOOLS  # weights + bias


def n_params() -> int:
    return N_PARAMS


def encode_obs(tile_map: np.ndarray) -> np.ndarray:
    """tile_map: (WORLD_H, WORLD_W) uint16 of tile IDs -> (N_CHAN, OBS_H, OBS_W) float32."""
    is_res = ((tile_map >= RES_RANGE[0]) & (tile_map <= RES_RANGE[1])).astype(np.float32)
    is_com = ((tile_map >= COM_RANGE[0]) & (tile_map <= COM_RANGE[1])).astype(np.float32)
    is_ind = ((tile_map >= IND_RANGE[0]) & (tile_map <= IND_RANGE[1])).astype(np.float32)
    is_inf = ((tile_map >= ROAD_RANGE[0]) & (tile_map <= ROAD_RANGE[1])).astype(np.float32)
    for tid in PLANT_TILES:
        is_inf = np.maximum(is_inf, (tile_map == tid).astype(np.float32))

    def pool(x):
        # Max-pool over OBS_STRIDE x OBS_STRIDE blocks
        return x.reshape(OBS_H, OBS_STRIDE, OBS_W, OBS_STRIDE).max(axis=(1, 3))

    return np.stack([pool(is_res), pool(is_com), pool(is_ind), pool(is_inf)], axis=0)


def _conv2d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Naive 3x3 SAME conv. x: (C_in, H, W), w: (C_out, C_in, 3, 3), b: (C_out,)."""
    pad = 1
    x_p = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    C_out = w.shape[0]
    H, W = x.shape[1], x.shape[2]
    out = np.empty((C_out, H, W), dtype=np.float32)
    for co in range(C_out):
        acc = np.zeros((H, W), dtype=np.float32)
        for ci in range(x.shape[0]):
            for ky in range(KSIZE):
                for kx in range(KSIZE):
                    acc += w[co, ci, ky, kx] * x_p[ci, ky:ky + H, kx:kx + W]
        out[co] = acc + b[co]
    return out


class ConvPolicy:
    """Single 3x3 conv: 4 input channels -> 20 output channels (one per tool)."""

    def __init__(self):
        self.w = np.zeros((N_TOOLS, N_CHAN, KSIZE, KSIZE), dtype=np.float32)
        self.b = np.zeros(N_TOOLS, dtype=np.float32)

    @staticmethod
    def param_count() -> int:
        return N_PARAMS

    def set_params(self, theta: np.ndarray) -> None:
        assert theta.shape == (N_PARAMS,), theta.shape
        n_w = N_TOOLS * N_CHAN * KSIZE * KSIZE
        self.w = theta[:n_w].reshape(N_TOOLS, N_CHAN, KSIZE, KSIZE).astype(np.float32)
        self.b = theta[n_w:].astype(np.float32)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w.ravel(), self.b.ravel()]).astype(np.float32)

    def reset(self, seed=None) -> None:
        """No-op for closed-loop policies (only ActionTape uses this)."""

    # Recommended hyperparams for CMA-ME with this policy
    SIGMA0 = 0.5
    RECOMMENDED_ES = "sep_cma_es"

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        """Return (tool, world_x, world_y) by greedy argmax over conv logits."""
        obs = encode_obs(tile_map)               # (N_CHAN, OBS_H, OBS_W)
        logits = _conv2d_same(obs, self.w, self.b)  # (N_TOOLS, OBS_H, OBS_W)
        flat = logits.reshape(-1)
        idx = int(np.argmax(flat))
        tool = idx // (OBS_H * OBS_W)
        rem = idx % (OBS_H * OBS_W)
        y_obs = rem // OBS_W
        x_obs = rem % OBS_W
        wx = x_obs * OBS_STRIDE + OBS_STRIDE // 2
        wy = y_obs * OBS_STRIDE + OBS_STRIDE // 2
        return tool, wx, wy


# ---------------------------------------------------------------------------
# ActionTape: open-loop fixed sequence of (tool, x, y) triples.
#
# The simplest possible policy. CMA-ME evolves a flat vector of 3 floats per
# step, mapped through tanh -> discrete index. Open-loop: ignores tile_map.
# Param count is n_actions * 3 (e.g. 100 actions = 300 params).
# ---------------------------------------------------------------------------

class ActionTape:
    SIGMA0 = 1.0                # bigger so initial population spreads action space
    RECOMMENDED_ES = "cma_es"   # full CMA fine at this param count

    def __init__(self, n_actions: int = 100):
        self.n_actions = n_actions
        self.theta = np.zeros((n_actions, 3), dtype=np.float32)
        self.step_idx = 0

    @staticmethod
    def param_count(n_actions: int = 100) -> int:
        return n_actions * 3

    def set_params(self, theta: np.ndarray) -> None:
        assert theta.shape == (self.n_actions * 3,), theta.shape
        self.theta = theta.reshape(self.n_actions, 3).astype(np.float32)

    def get_params(self) -> np.ndarray:
        return self.theta.ravel().astype(np.float32)

    def reset(self, seed=None) -> None:
        self.step_idx = 0

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        # Cycle if we run out of actions (shouldn't happen if n_actions matches rollout length).
        i = self.step_idx % self.n_actions
        self.step_idx += 1
        t, x, y = self.theta[i]
        # tanh -> [0, 1] -> discrete index. theta=0 maps to the middle of the range.
        tool = min(int((np.tanh(t) + 1.0) * 0.5 * N_TOOLS), N_TOOLS - 1)
        wx   = min(int((np.tanh(x) + 1.0) * 0.5 * WORLD_W), WORLD_W - 1)
        wy   = min(int((np.tanh(y) + 1.0) * 0.5 * WORLD_H), WORLD_H - 1)
        return tool, wx, wy


# ---------------------------------------------------------------------------
# ContextualTape: ActionTape + a small shared linear "delta" off the current
# map summary. Cheapest possible closed-loop variant of the tape.
#
# At step i:
#   tanh-action[i] = tanh( base[i] + W @ features(map) )
# where:
#   base is (n_actions, 3) like ActionTape
#   features(map) is a fixed N_FEAT-dim normalized summary of the current map
#   W is a SHARED (3, N_FEAT) matrix used for every step
#
# Param count: n_actions*3 + 3*N_FEAT  (e.g. 100*3 + 3*10 = 330).
# Only 30 more params than ActionTape, but the policy can react to what it's
# already built. If the open-loop tape wins, the search just learns W ~ 0.
# ---------------------------------------------------------------------------

_CTX_N_FEAT = 10  # see _ctx_features() below


def _ctx_features(tile_map: np.ndarray, step: int, n_actions: int) -> np.ndarray:
    """Tiny normalized summary of the current map state.

    Returns a length-N_FEAT vector in roughly [0, 1] each. The features are
    deliberately coarse so the search direction in W-space is well-conditioned.
    """
    H, W = tile_map.shape
    total = float(H * W)
    is_res = (tile_map >= RES_RANGE[0]) & (tile_map <= RES_RANGE[1])
    is_com = (tile_map >= COM_RANGE[0]) & (tile_map <= COM_RANGE[1])
    is_ind = (tile_map >= IND_RANGE[0]) & (tile_map <= IND_RANGE[1])
    is_road = (tile_map >= ROAD_RANGE[0]) & (tile_map <= ROAD_RANGE[1])
    is_plant = np.isin(tile_map, list(PLANT_TILES))
    is_empty = tile_map == 0
    built_mask = is_res | is_com | is_ind | is_road | is_plant
    feats = np.array([
        is_res.sum() / total,
        is_com.sum() / total,
        is_ind.sum() / total,
        is_road.sum() / total,
        is_plant.sum() / total,
        is_empty.sum() / total,
        built_mask.sum() / total,
        # Centroid of built tiles (or 0.5 if none) — gives a sense of where stuff is.
        (np.argwhere(built_mask)[:, 1].mean() / W) if built_mask.any() else 0.5,
        (np.argwhere(built_mask)[:, 0].mean() / H) if built_mask.any() else 0.5,
        step / max(1, n_actions),  # normalized progress through the rollout
    ], dtype=np.float32)
    return feats


class ContextualTape:
    SIGMA0 = 1.0
    RECOMMENDED_ES = "sep_cma_es"  # 330 params is on the edge of full-CMA's comfort zone

    def __init__(self, n_actions: int = 100, n_features: int = _CTX_N_FEAT):
        self.n_actions = n_actions
        self.n_features = n_features
        self.base = np.zeros((n_actions, 3), dtype=np.float32)
        self.W = np.zeros((3, n_features), dtype=np.float32)
        self.step_idx = 0

    @staticmethod
    def param_count(n_actions: int = 100, n_features: int = _CTX_N_FEAT) -> int:
        return n_actions * 3 + 3 * n_features

    def set_params(self, theta: np.ndarray) -> None:
        n_base = self.n_actions * 3
        n_w = 3 * self.n_features
        assert theta.shape == (n_base + n_w,), theta.shape
        self.base = theta[:n_base].reshape(self.n_actions, 3).astype(np.float32)
        self.W = theta[n_base:n_base + n_w].reshape(3, self.n_features).astype(np.float32)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.base.ravel(), self.W.ravel()]).astype(np.float32)

    def reset(self, seed=None) -> None:
        self.step_idx = 0

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        i = self.step_idx % self.n_actions
        feats = _ctx_features(tile_map, self.step_idx, self.n_actions)
        delta = self.W @ feats                       # shape (3,)
        t, x, y = self.base[i] + delta
        self.step_idx += 1
        tool = min(int((np.tanh(t) + 1.0) * 0.5 * N_TOOLS), N_TOOLS - 1)
        wx   = min(int((np.tanh(x) + 1.0) * 0.5 * WORLD_W), WORLD_W - 1)
        wy   = min(int((np.tanh(y) + 1.0) * 0.5 * WORLD_H), WORLD_H - 1)
        return tool, wx, wy


# ---------------------------------------------------------------------------
# MLPPolicy: flatten obs -> ReLU -> factored (tool, position) heads.
#
# Closed-loop. Output is factored: tool comes from one softmax over N_TOOLS,
# position comes from one softmax over OBS_H*OBS_W cells (snapped to block
# centers like ConvPolicy). Independent argmaxes.
# ---------------------------------------------------------------------------

class MLPPolicy:
    SIGMA0 = 0.1
    RECOMMENDED_ES = "sep_cma_es"

    def __init__(self, hidden: int = 32):
        self.hidden = hidden
        self._in_dim = N_CHAN * OBS_H * OBS_W
        self._out_pos_dim = OBS_H * OBS_W
        # Layer 1: in -> hidden
        # Layer 2a: hidden -> N_TOOLS (tool head)
        # Layer 2b: hidden -> OBS_H*OBS_W (pos head)
        self.W1 = np.zeros((hidden, self._in_dim), dtype=np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W_tool = np.zeros((N_TOOLS, hidden), dtype=np.float32)
        self.b_tool = np.zeros(N_TOOLS, dtype=np.float32)
        self.W_pos = np.zeros((self._out_pos_dim, hidden), dtype=np.float32)
        self.b_pos = np.zeros(self._out_pos_dim, dtype=np.float32)

    @classmethod
    def param_count(cls, hidden: int = 32) -> int:
        in_dim = N_CHAN * OBS_H * OBS_W
        out_pos = OBS_H * OBS_W
        return (
            hidden * in_dim + hidden +
            N_TOOLS * hidden + N_TOOLS +
            out_pos * hidden + out_pos
        )

    def set_params(self, theta: np.ndarray) -> None:
        h, in_dim, out_pos = self.hidden, self._in_dim, self._out_pos_dim
        off = 0
        self.W1 = theta[off:off + h * in_dim].reshape(h, in_dim).astype(np.float32); off += h * in_dim
        self.b1 = theta[off:off + h].astype(np.float32); off += h
        self.W_tool = theta[off:off + N_TOOLS * h].reshape(N_TOOLS, h).astype(np.float32); off += N_TOOLS * h
        self.b_tool = theta[off:off + N_TOOLS].astype(np.float32); off += N_TOOLS
        self.W_pos = theta[off:off + out_pos * h].reshape(out_pos, h).astype(np.float32); off += out_pos * h
        self.b_pos = theta[off:off + out_pos].astype(np.float32); off += out_pos

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W_tool.ravel(), self.b_tool,
            self.W_pos.ravel(), self.b_pos,
        ]).astype(np.float32)

    def reset(self, seed=None) -> None:
        pass

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        obs = encode_obs(tile_map).ravel()
        h = np.maximum(0.0, self.W1 @ obs + self.b1)   # ReLU
        tool_logits = self.W_tool @ h + self.b_tool
        pos_logits  = self.W_pos  @ h + self.b_pos
        tool = int(np.argmax(tool_logits))
        pos_idx = int(np.argmax(pos_logits))
        y_obs = pos_idx // OBS_W
        x_obs = pos_idx %  OBS_W
        wx = x_obs * OBS_STRIDE + OBS_STRIDE // 2
        wy = y_obs * OBS_STRIDE + OBS_STRIDE // 2
        return tool, wx, wy


# ---------------------------------------------------------------------------
# DeepConvPolicy: stack of 3x3 conv layers with ReLU, ending in N_TOOLS chans.
# Same output decoding as ConvPolicy (joint argmax over 20x20x24 logits).
# ---------------------------------------------------------------------------

class DeepConvPolicy:
    SIGMA0 = 0.2
    RECOMMENDED_ES = "sep_cma_es"

    def __init__(self, channels: tuple[int, ...] = (16, 32)):
        self.channels = tuple(channels)
        layers = []
        prev = N_CHAN
        for c in self.channels:
            layers.append((prev, c))
            prev = c
        layers.append((prev, N_TOOLS))
        self._layer_shapes = layers
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        # Initialize zeros — set_params replaces.
        for cin, cout in layers:
            self._weights.append(np.zeros((cout, cin, KSIZE, KSIZE), dtype=np.float32))
            self._biases.append(np.zeros(cout, dtype=np.float32))

    @classmethod
    def param_count(cls, channels: tuple[int, ...] = (16, 32)) -> int:
        layers = []
        prev = N_CHAN
        for c in channels:
            layers.append((prev, c))
            prev = c
        layers.append((prev, N_TOOLS))
        return sum(cout * cin * KSIZE * KSIZE + cout for cin, cout in layers)

    def set_params(self, theta: np.ndarray) -> None:
        off = 0
        new_w, new_b = [], []
        for cin, cout in self._layer_shapes:
            n_w = cout * cin * KSIZE * KSIZE
            new_w.append(theta[off:off + n_w].reshape(cout, cin, KSIZE, KSIZE).astype(np.float32))
            off += n_w
            new_b.append(theta[off:off + cout].astype(np.float32))
            off += cout
        self._weights = new_w
        self._biases = new_b

    def get_params(self) -> np.ndarray:
        parts = []
        for w, b in zip(self._weights, self._biases):
            parts.append(w.ravel()); parts.append(b)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None) -> None:
        pass

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        x = encode_obs(tile_map)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = _conv2d_same(x, w, b)
            # ReLU on all except the last (logits) layer
            if i < len(self._weights) - 1:
                x = np.maximum(0.0, x)
        flat = x.reshape(-1)
        idx = int(np.argmax(flat))
        tool = idx // (OBS_H * OBS_W)
        rem = idx % (OBS_H * OBS_W)
        y_obs = rem // OBS_W
        x_obs = rem % OBS_W
        wx = x_obs * OBS_STRIDE + OBS_STRIDE // 2
        wy = y_obs * OBS_STRIDE + OBS_STRIDE // 2
        return tool, wx, wy


# ---------------------------------------------------------------------------
# HybridPolicy: open-loop ActionTape for the first n_tape steps, then a
# closed-loop DeepConvPolicy for the remaining steps.
#
# Motivation: pure ActionTape wins by carpet-bombing the map — place enough
# diverse stuff that a few zones randomly end up adjacent to a power source
# and grow. But it's purely open-loop and wastes most placements. A pure
# closed-loop net fails because from an empty map, its zero-init output is
# degenerate and CMA-ES never breaks the symmetry.
#
# The hybrid is a tuple <tape params, net weights>. The tape solves the
# bootstrap (puts enough stuff on the map that the net has something to
# react to). The net then refines, placing infrastructure where it sees
# zones that need power/roads/etc.
#
# Single flat θ for CMA-ES: [tape (n_tape*3), net (DeepConv.param_count)].
# Episode length is set by the env's n_actions; first n_tape go to the
# tape, the rest go to the net.
# ---------------------------------------------------------------------------

class HybridPolicy:
    SIGMA0 = 0.3                  # compromise between tape (1.0) and net (~0.1-0.2)
    RECOMMENDED_ES = "sep_cma_es"  # ~thousands of params

    def __init__(self, n_tape: int = 50, channels: tuple[int, ...] = (8, 16)):
        self.n_tape = n_tape
        self.channels = tuple(channels)
        self.tape = ActionTape(n_actions=n_tape)
        self.net = DeepConvPolicy(channels=channels)
        self.step_idx = 0

    @classmethod
    def param_count(cls, n_tape: int = 50, channels: tuple[int, ...] = (8, 16)) -> int:
        return (
            ActionTape.param_count(n_actions=n_tape)
            + DeepConvPolicy.param_count(channels=channels)
        )

    def set_params(self, theta: np.ndarray) -> None:
        n_tape_params = self.n_tape * 3
        self.tape.set_params(theta[:n_tape_params])
        self.net.set_params(theta[n_tape_params:])

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.tape.get_params(), self.net.get_params()]).astype(np.float32)

    def reset(self, seed=None) -> None:
        self.step_idx = 0
        self.tape.reset()
        self.net.reset()

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        if self.step_idx < self.n_tape:
            self.step_idx += 1
            return self.tape.act(tile_map)
        self.step_idx += 1
        return self.net.act(tile_map)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RandomPrefixPolicy: first n_random actions are *stochastic* random draws
# (different per eval seed), then DeepConvPolicy for the rest.
#
# Motivation: HybridPolicy showed that an evolved-tape prefix doesn't help
# the net beyond what tape alone produces. The hypothesis here is that
# *random* scaffolding is even better: it forces the net to be robust across
# different initial map states, and combined with multi-eval averaging
# (--n-evals N in qd_train) it gives CMA-ME a smoothed gradient — policies
# near a fitness peak benefit even when individual rollouts are noisy.
#
# Only the net is parameterized; the random prefix has no learnable params.
# At the same total param count as a pure DeepConvPolicy, but the net is
# trained against varied starting states instead of pristine empty maps.
# ---------------------------------------------------------------------------

# Same weighted tool pool as random_smoke.py — skews toward useful tools,
# keeps destructive options (BULLDOZER, QUERY) out of the random prefix.
_RANDOM_PREFIX_POOL = [
    (Tool.RESIDENTIAL,   8),
    (Tool.COMMERCIAL,    4),
    (Tool.INDUSTRIAL,    4),
    (Tool.ROAD,         10),
    (Tool.WIRE,          4),
    (Tool.PARK,          2),
    (Tool.RAILROAD,      1),
    (Tool.COALPOWER,     2),   # added — power source matters a lot
]


class RandomPrefixPolicy:
    SIGMA0 = 0.2
    RECOMMENDED_ES = "sep_cma_es"

    def __init__(self, n_random: int = 50, channels: tuple[int, ...] = (16, 32)):
        self.n_random = n_random
        self.channels = tuple(channels)
        self.net = DeepConvPolicy(channels=channels)
        self.step_idx = 0
        self._rng: np.random.Generator | None = None

    @classmethod
    def param_count(cls, n_random: int = 50, channels: tuple[int, ...] = (16, 32)) -> int:
        return DeepConvPolicy.param_count(channels=channels)

    def set_params(self, theta: np.ndarray) -> None:
        self.net.set_params(theta)

    def get_params(self) -> np.ndarray:
        return self.net.get_params()

    def reset(self, seed=None) -> None:
        self.step_idx = 0
        self.net.reset()
        # Seed the prefix RNG. If no seed is passed, fall back to entropy.
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        if self.step_idx < self.n_random:
            self.step_idx += 1
            tools, weights = zip(*_RANDOM_PREFIX_POOL)
            p = np.array(weights, dtype=np.float64)
            p = p / p.sum()
            tool = int(self._rng.choice(tools, p=p))
            wx = int(self._rng.integers(0, WORLD_W))
            wy = int(self._rng.integers(0, WORLD_H))
            return tool, wx, wy
        self.step_idx += 1
        return self.net.act(tile_map)


# ---------------------------------------------------------------------------
# LayoutGenome: the genome IS the city, no sequential policy at all.
#
# A flat float vector decodes into a (GRID_H, GRID_W) categorical grid of
# zone types plus a single tax-rate gene. At build time we:
#   1. clearMap
#   2. lay a regular WIRE grid (every CELL_SIZE-th row and column) so every
#      zone in the grid is adjacent to a power conductor
#   3. drop a fixed coal plant (power source)
#   4. set city tax from the gene
#   5. for each grid cell, place the chosen zone at the cell center
#   6. tick the engine for STABILIZATION_TICKS so zones can grow
#
# Why wires, not roads: in this Micropolis engine, plain road tiles do
# NOT have CONDBIT, so they don't propagate power. A wire grid does both
# jobs — provides power and (since this engine doesn't require strict
# road adjacency for zone growth) lets zones develop. Empirically a wire
# grid + zones produces ~21 grown residential zones and cityPop~440;
# the same shape with roads instead of wires produces 0.
#
# We integrate as a "policy" by exposing the standard set_params /
# param_count / reset / act surface AND an IS_LAYOUT class flag plus a
# build(env) method. evaluate._evaluate_once special-cases IS_LAYOUT to
# call build() once and skip the action loop.
#
# This setup directly asks the simulator "given a viable power grid and
# guaranteed power source, which zone composition do you reward?" —
# more interpretable than evolving an open-ended action sequence.
# ---------------------------------------------------------------------------

# Zone vocabulary. The simulator does NOT reward placing roads/wires inside
# the grid (those are background); the search is purely over zone composition.
LAYOUT_CATEGORIES: tuple = (
    None,                # 0: empty
    Tool.RESIDENTIAL,    # 1
    Tool.COMMERCIAL,     # 2
    Tool.INDUSTRIAL,     # 3
    Tool.PARK,           # 4
)


class LayoutGenome:
    SIGMA0 = 0.5
    RECOMMENDED_ES = "sep_cma_es"
    IS_LAYOUT = True

    # Defaults; can be overridden per-instance via __init__ kwargs.
    # CELL_SIZE = 4 chosen carefully: roads at cols 0, 4, 8, ... and the
    # 3x3 zone center at col 4i+2 puts the zone's leftmost column (4i+1)
    # adjacent to the road at col 4i AND rightmost column (4i+3) adjacent
    # to the road at col 4(i+1). Same for rows. So every zone has road
    # access on all four cardinal sides — the Micropolis precondition for
    # growth. Larger cell sizes leave zones marooned (no adjacency).
    GRID_H = 25          # 100 / 4
    GRID_W = 30          # 120 / 4
    CELL_SIZE = 4
    N_CAT = len(LAYOUT_CATEGORIES)

    # Fixed coal plant location (top-center of the map). Power propagates
    # through the road grid, so location doesn't really matter — pick a
    # consistent spot so genomes don't have to evolve plant placement.
    PLANT_XY: tuple[int, int] = (WORLD_W // 2, 4)

    # Max tax value the engine accepts is 20% — anything beyond saturates.
    MAX_TAX = 20

    def __init__(self, grid_h: int | None = None, grid_w: int | None = None,
                 cell_size: int | None = None):
        if grid_h is not None: self.GRID_H = grid_h
        if grid_w is not None: self.GRID_W = grid_w
        if cell_size is not None: self.CELL_SIZE = cell_size
        self.theta = np.zeros(self.param_count(
            grid_h=self.GRID_H, grid_w=self.GRID_W
        ), dtype=np.float32)

    @classmethod
    def param_count(cls, grid_h: int | None = None, grid_w: int | None = None,
                    cell_size: int | None = None) -> int:
        gh = grid_h if grid_h is not None else cls.GRID_H
        gw = grid_w if grid_w is not None else cls.GRID_W
        return gh * gw * cls.N_CAT + 1   # +1 for tax-rate gene

    def set_params(self, theta: np.ndarray) -> None:
        assert theta.shape == (self.param_count(grid_h=self.GRID_H, grid_w=self.GRID_W),), \
            f"expected {self.param_count(grid_h=self.GRID_H, grid_w=self.GRID_W)} params, got {theta.shape}"
        self.theta = theta.astype(np.float32)

    def get_params(self) -> np.ndarray:
        return self.theta.copy()

    def reset(self, seed: int | None = None) -> None:
        # Stateless across rollouts; nothing to reset.
        pass

    def act(self, tile_map: np.ndarray) -> tuple[int, int, int]:
        # Should never be called (IS_LAYOUT short-circuits the action loop).
        raise RuntimeError("LayoutGenome.act() should not be called — IS_LAYOUT=True")

    # ----- decoding -----

    def decode(self) -> tuple[np.ndarray, int]:
        """Return (categorical (GRID_H, GRID_W) array, tax_rate)."""
        n_grid = self.GRID_H * self.GRID_W * self.N_CAT
        logits = self.theta[:n_grid].reshape(self.GRID_H, self.GRID_W, self.N_CAT)
        grid = logits.argmax(axis=-1).astype(np.int32)
        tax_scalar = float(self.theta[n_grid])
        tax = int((np.tanh(tax_scalar) + 1.0) * 0.5 * self.MAX_TAX)
        return grid, max(0, min(self.MAX_TAX, tax))

    # ----- building -----

    def build(self, env) -> None:
        """Place tiles into a freshly-cleared map. Caller stabilizes via env.tick()."""
        eng = env.engine
        eng.clearMap()
        grid, tax = self.decode()
        eng.setCityTax(tax)

        # Background wire grid — every CELL_SIZE-th row and column.
        # Wires conduct power (CONDBIT) where plain roads don't, so this
        # is what makes zones eligible for growth. See variant A in
        # /tmp/grid_diag.py for the empirical comparison.
        for x in range(0, WORLD_W, self.CELL_SIZE):
            for y in range(WORLD_H):
                env.place(Tool.WIRE, x, y)
        for y in range(0, WORLD_H, self.CELL_SIZE):
            for x in range(WORLD_W):
                env.place(Tool.WIRE, x, y)

        # Guaranteed power source — the genome doesn't have to find one.
        env.place(Tool.COALPOWER, *self.PLANT_XY)

        # Drop each zone at its cell center.
        half = self.CELL_SIZE // 2
        for j in range(self.GRID_H):
            for i in range(self.GRID_W):
                cat = int(grid[j, i])
                tool = LAYOUT_CATEGORIES[cat]
                if tool is None:
                    continue
                wx = i * self.CELL_SIZE + half
                wy = j * self.CELL_SIZE + half
                env.place(tool, wx, wy)


POLICY_REGISTRY = {
    "conv":      ConvPolicy,
    "tape":      ActionTape,
    "ctxtape":   ContextualTape,
    "mlp":       MLPPolicy,
    "deepconv":  DeepConvPolicy,
    "hybrid":    HybridPolicy,
    "randprefix": RandomPrefixPolicy,
    "layout":    LayoutGenome,
}


def make_policy(name: str, **kwargs):
    """Construct a policy by name. Unknown kwargs are passed to the policy ctor."""
    if name not in POLICY_REGISTRY:
        raise KeyError(f"unknown policy {name!r}; choices: {sorted(POLICY_REGISTRY)}")
    cls = POLICY_REGISTRY[name]
    return cls(**kwargs)


def policy_param_count(name: str, **kwargs) -> int:
    """Param count for a policy without constructing one (for CMA-ES setup)."""
    cls = POLICY_REGISTRY[name]
    # ConvPolicy.param_count() takes no args; others may accept their ctor kwargs.
    try:
        return cls.param_count(**kwargs)
    except TypeError:
        return cls.param_count()
