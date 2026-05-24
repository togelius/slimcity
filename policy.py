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
