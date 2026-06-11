from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int]


def load_txt_map(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if " " in line or "," in line:
            tokens = line.replace(",", " ").split()
            rows.append([int(x) for x in tokens])
        else:
            rows.append([int(ch) for ch in line])
    if not rows:
        raise ValueError(f"No map data found in {path}.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All map rows must have the same width.")
    return np.asarray(rows, dtype=np.int8)


def create_demo_map(rows: int = 30, cols: int = 45) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=np.int8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    exit_col = cols // 2
    grid[0, exit_col] = 0
    grid[1, exit_col] = 0
    grid[rows // 2, 5 : cols - 5] = 1
    grid[rows // 2, cols // 2 - 2 : cols // 2 + 3] = 0
    grid[7 : rows - 6, cols // 3] = 1
    grid[rows // 3, cols // 3 - 2 : cols // 3 + 3] = 0
    grid[6 : rows - 4, 2 * cols // 3] = 1
    grid[2 * rows // 3, 2 * cols // 3 - 2 : 2 * cols // 3 + 3] = 0
    return grid


def parse_points(text: str | None) -> List[GridPoint]:
    if not text:
        return []
    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        row, col = item.replace(",", " ").split()[:2]
        points.append((int(row), int(col)))
    return points


def find_border_exits(grid: np.ndarray) -> List[GridPoint]:
    rows, cols = grid.shape
    exits: List[GridPoint] = []
    for c in range(cols):
        for r in (0, rows - 1):
            if grid[r, c] == 0:
                exits.append((r, c))
    for r in range(1, rows - 1):
        for c in (0, cols - 1):
            if grid[r, c] == 0:
                exits.append((r, c))
    if exits:
        return exits
    fallback = np.argwhere(grid == 0)
    if len(fallback) == 0:
        raise ValueError("Map has no free cells.")
    point = fallback[-1]
    return [(int(point[0]), int(point[1]))]


def choose_default_starts(grid: np.ndarray, exits: Sequence[GridPoint], n_agents: int) -> List[GridPoint]:
    free = np.argwhere(grid == 0)
    exits_arr = np.asarray(exits, dtype=np.int32)
    scored = []
    for point in free:
        if any(np.array_equal(point, exit_point) for exit_point in exits_arr):
            continue
        dist = np.min(np.linalg.norm(exits_arr - point, axis=1))
        scored.append((float(dist), (int(point[0]), int(point[1]))))
    scored.sort(reverse=True, key=lambda item: item[0])
    if len(scored) < n_agents:
        raise ValueError("Not enough free cells to place all agents.")
    return [point for _, point in scored[:n_agents]]


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
