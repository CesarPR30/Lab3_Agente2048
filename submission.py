# solucion.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple
import math
import numpy as np


Action = str  # "up", "down", "left", "right"
OPP = {"up":"down","down":"up","left":"right","right":"left"}

@dataclass
class Agent:
    seed: int | None = None
    depth: int = 3
    sample_chance: int = 4
    last_action: str | None = None

    def act(self, board, legal_actions) -> str:
        b = np.asarray(board, dtype=np.int64)

        best_a = legal_actions[0]
        best_v = -1e18

        for a in legal_actions:
            # penaliza el opuesto inmediato
            backtrack_pen = 0.15 if (self.last_action and a == OPP.get(self.last_action)) else 0.0

            nb, moved, reward = _apply_move(b, a)
            if not moved:
                continue
            v = reward + self._expectimax(nb, self.depth - 1, b.shape[0]) - backtrack_pen

            if v > best_v:
                best_v = v
                best_a = a

        self.last_action = best_a
        return best_a

    def _expectimax(self, board: np.ndarray, depth: int, size: int) -> float:
        key = _pack(board)
        return _exp_cached(key, depth, size, self.sample_chance)

def _unpack(packed: Tuple[int, ...], size: int) -> np.ndarray:
    arr = np.array(packed, dtype=np.int64).reshape(size, size)
    return arr

def _pack(board: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in board.ravel())

@lru_cache(maxsize=200000)
def _exp_cached(packed: Tuple[int, ...], depth: int, size: int, sample_chance: int) -> float:
    board = _unpack(packed, size)

    if depth <= 0:
        return _heuristic(board)

    empties = np.argwhere(board == 0)
    if empties.size == 0:
        return _max_node(board, depth, size, sample_chance)
    
    n_empty = len(empties)
    if n_empty > sample_chance:
        idxs = np.linspace(0, n_empty - 1, sample_chance).astype(int)
        empties = empties[idxs]
        n_empty = len(empties)

    p_cell = 1.0 / n_empty
    val = 0.0

    for (r, c) in empties:
        b2 = board.copy()
        b2[r, c] = 2
        val += p_cell * 0.9 * _max_node(b2, depth - 1, size, sample_chance)
        b4 = board.copy()
        b4[r, c] = 4
        val += p_cell * 0.1 * _max_node(b4, depth - 1, size, sample_chance)

    return val

def _max_node(board: np.ndarray, depth: int, size: int, sample_chance: int) -> float:
    best = -1e18
    any_moved = False
    for a in ("up", "down", "left", "right"):
        nb, moved, reward = _apply_move(board, a)
        if not moved:
            continue
        any_moved = True
        v = reward + _exp_cached(_pack(nb), depth, size, sample_chance)
        if v > best:
            best = v

    if not any_moved:
        return _heuristic(board) - 1e6
    return best


def _apply_move(board: np.ndarray, action: Action) -> Tuple[np.ndarray, bool, float]:
    b = board.copy()
    if action == "left":
        nb, reward = _move_left(b)
    elif action == "right":
        nb, reward = _move_right(b)
    elif action == "up":
        nb, reward = _move_up(b)
    elif action == "down":
        nb, reward = _move_down(b)
    else:
        raise ValueError(f"Unknown action: {action}")

    moved = not np.array_equal(nb, board)
    return nb, moved, float(reward)

def _compress_and_merge_row(row: np.ndarray) -> Tuple[np.ndarray, int]:
    tiles = row[row != 0].tolist()
    merged = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            merged.append(v)
            reward += v
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (len(row) - len(merged))
    return np.array(merged, dtype=np.int64), reward

def _move_left(board: np.ndarray) -> Tuple[np.ndarray, int]:
    size = board.shape[0]
    out = np.zeros_like(board)
    reward = 0
    for r in range(size):
        new_row, rew = _compress_and_merge_row(board[r, :])
        out[r, :] = new_row
        reward += rew
    return out, reward

def _move_right(board: np.ndarray) -> Tuple[np.ndarray, int]:
    rev = np.fliplr(board)
    moved, reward = _move_left(rev)
    return np.fliplr(moved), reward

def _move_up(board: np.ndarray) -> Tuple[np.ndarray, int]:
    tr = board.T
    moved, reward = _move_left(tr)
    return moved.T, reward

def _move_down(board: np.ndarray) -> Tuple[np.ndarray, int]:
    tr = board.T
    moved, reward = _move_right(tr)
    return moved.T, reward


def _heuristic(board: np.ndarray) -> float:
    b = board.astype(np.float64)
    empty = float(np.sum(b == 0))

    max_tile = float(b.max()) if b.size else 0.0
    log_max = math.log2(max_tile) if max_tile > 0 else 0.0

    smooth = -_smoothness(b)          
    mono = _monotonicity(b)       

    snake = _snake_weight(b)
    corner = _corner_bonus_fixed(b)

    return (
        6.0 * empty +
        2.0 * smooth +
        2.0 * mono +
        3 * corner +
        1.0 * log_max +
        0.0015 * snake
    )

def _smoothness(b: np.ndarray) -> float:
    size = b.shape[0]
    def l2(x):
        return math.log2(x) if x > 0 else 0.0

    s = 0.0
    for r in range(size):
        for c in range(size):
            v = b[r, c]
            if v <= 0:
                continue
            lv = l2(v)
            if c + 1 < size and b[r, c + 1] > 0:
                s += abs(lv - l2(b[r, c + 1]))
            if r + 1 < size and b[r + 1, c] > 0:
                s += abs(lv - l2(b[r + 1, c]))
    return s

def _monotonicity(b: np.ndarray) -> float:
    size = b.shape[0]
    def logs(arr):
        out = np.zeros_like(arr, dtype=np.float64)
        nz = arr > 0
        out[nz] = np.log2(arr[nz])
        return out

    score = 0.0
    for r in range(size):
        row = logs(b[r, :])
        inc = np.sum(np.maximum(0.0, row[1:] - row[:-1]))
        dec = np.sum(np.maximum(0.0, row[:-1] - row[1:]))
        score += max(inc, dec)
    for c in range(size):
        col = logs(b[:, c])
        inc = np.sum(np.maximum(0.0, col[1:] - col[:-1]))
        dec = np.sum(np.maximum(0.0, col[:-1] - col[1:]))
        score += max(inc, dec)

    return score

def _max_in_corner(b: np.ndarray) -> float:
    m = b.max()
    corners = [b[0,0], b[0,-1], b[-1,0], b[-1,-1]]
    return 1.0 if any(x == m for x in corners) and m > 0 else 0.0

def _snake_weight(b: np.ndarray) -> float:
    W = np.array([
        [65536, 32768, 16384,  8192],
        [  512,  1024,  2048,  4096],
        [  256,   128,    64,    32],
        [    2,     4,     8,    16],
    ], dtype=np.float64)

    x = b.copy().astype(np.float64)
    nz = x > 0
    x[nz] = np.log2(x[nz])
    return float(np.sum(W * x))

def _corner_bonus_fixed(b: np.ndarray) -> float:
    m = b.max()
    return 1.0 if b[0,0] == m and m > 0 else 0.0