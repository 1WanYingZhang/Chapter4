from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_paths(grid_map: np.ndarray, paths: Sequence[Sequence[np.ndarray]], exits, save_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_map, cmap="Greys", origin="upper")

    for exit_point in exits:
        ax.scatter(exit_point[1], exit_point[0], marker="s", s=100, c="limegreen", edgecolor="black")

    cmap = plt.get_cmap("tab20")
    for i, path in enumerate(paths):
        arr = np.asarray(path)
        if len(arr) == 0:
            continue
        ax.plot(arr[:, 1], arr[:, 0], color=cmap(i % 20), linewidth=1.8, label=f"agent_{i}")
        ax.scatter(arr[0, 1], arr[0, 0], color=cmap(i % 20), marker="o", s=35)
        ax.scatter(arr[-1, 1], arr[-1, 0], color=cmap(i % 20), marker="x", s=45)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("MAPPO multi-agent evacuation paths")
    if len(paths) <= 12:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(csv_path: str | Path, save_path: str | Path) -> None:
    import csv
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["episode_reward"]) for row in rows]
    evacuated = [int(row["num_evacuated"]) for row in rows]
    steps = [int(row["episode_steps"]) for row in rows]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(episodes, rewards)
    axes[0].set_ylabel("reward")
    axes[1].plot(episodes, evacuated)
    axes[1].set_ylabel("evacuated")
    axes[2].plot(episodes, steps)
    axes[2].set_ylabel("steps")
    axes[2].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def animate_paths(
    grid_map: np.ndarray,
    paths: Sequence[Sequence[np.ndarray]],
    exits,
    save_path: str | Path,
    fps: int = 8,
    trail: bool = True,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    padded_paths = _pad_paths(paths)
    n_agents, n_frames, _ = padded_paths.shape

    aspect = grid_map.shape[1] / max(1, grid_map.shape[0])
    fig_width = min(12.0, max(6.0, 7.0 * aspect))
    fig_height = min(12.0, max(6.0, 7.0 / max(aspect, 0.4)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(grid_map, cmap="Greys", origin="upper")

    for exit_point in exits:
        ax.scatter(exit_point[1], exit_point[0], marker="s", s=100, c="limegreen", edgecolor="black")

    cmap = plt.get_cmap("tab20")
    lines = []
    markers = []
    labels = []
    for i in range(n_agents):
        color = cmap(i % 20)
        line, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.85)
        marker = ax.scatter([], [], color=color, marker="o", s=45, edgecolor="black", linewidth=0.4)
        lines.append(line)
        markers.append(marker)
        if n_agents <= 20:
            labels.append(ax.text(0, 0, str(i), color="black", fontsize=8, ha="center", va="center"))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, grid_map.shape[1] - 0.5)
    ax.set_ylim(grid_map.shape[0] - 0.5, -0.5)

    def update(frame: int):
        artists = []
        for i in range(n_agents):
            segment = padded_paths[i, : frame + 1]
            current = segment[-1]
            if trail:
                lines[i].set_data(segment[:, 1], segment[:, 0])
            else:
                lines[i].set_data([], [])
            markers[i].set_offsets([[current[1], current[0]]])
            artists.extend([lines[i], markers[i]])
            if labels:
                labels[i].set_position((current[1], current[0]))
                artists.append(labels[i])
        ax.set_title(f"MAPPO evacuation step {frame}/{n_frames - 1}")
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / max(1, fps), blit=False)
    actual_path = save_path
    try:
        if save_path.suffix.lower() == ".mp4":
            anim.save(save_path, writer=FFMpegWriter(fps=fps), dpi=160)
        else:
            if save_path.suffix.lower() != ".gif":
                actual_path = save_path.with_suffix(".gif")
            anim.save(actual_path, writer=PillowWriter(fps=fps), dpi=140)
    except Exception:
        actual_path = save_path.with_suffix(".gif")
        anim.save(actual_path, writer=PillowWriter(fps=fps), dpi=140)
    finally:
        plt.close(fig)
    return actual_path


def _pad_paths(paths: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    arrays = []
    max_len = max((len(path) for path in paths), default=1)
    for path in paths:
        arr = np.asarray(path, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1, 2), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        if len(arr) < max_len:
            pad = np.repeat(arr[-1][None, :], max_len - len(arr), axis=0)
            arr = np.vstack([arr, pad])
        arrays.append(arr)
    if not arrays:
        arrays.append(np.zeros((max_len, 2), dtype=np.float32))
    return np.stack(arrays, axis=0)
