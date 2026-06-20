from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


GRID_TICK_STEP = 5
OBSTACLE_COLOR = "#727272"
EXIT_COLOR = "green"
GRID_LINE_COLOR = "black"
SHIP_FREE_COLOR = "#abd4f0"
SHIP_OBSTACLE_COLOR = "white"
SHIP_GRID_LINE_COLOR = "#a0a0a0"
SHIP_MAP_NAME = "邮轮地图.txt"
SHIP_IMAGE_NAME = "D11.png"
SHIP_TX1 = 15
SHIP_TY1 = 255
SHIP_XYDAIM = 5
SHIP_XMIN = -4
SHIP_YMIN = -52
CONGESTION_HEATMAP_COLORS = [
    "#7374a8",
    "#5c92b0",
    "#55b99f",
    "#c8df92",
    "#f1d38b",
    "#e9a7b5",
]
AGENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#f2b701",
    "#aec7e8",
    "#ffbb78",
    "#c5b0d5",
    "#ff9896",
    "#c49c94",
    "#f7b6d2",
    "#9edae5",
    "#f77f00",
    "#3a0ca3",
    "#fb6f92",
    "#00a6fb",
    "#9b2226",
]


def _congestion_heatmap_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("congestion_stay_time", CONGESTION_HEATMAP_COLORS, N=256)


def _agent_color(agent_id: int) -> str:
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


def _grid_ticks(length: int, step: int = GRID_TICK_STEP) -> np.ndarray:
    return np.arange(0, length, step, dtype=int)


def _set_grid_axis_ticks(ax, grid_shape: tuple[int, int], step: int = GRID_TICK_STEP) -> None:
    rows, cols = grid_shape
    x_ticks = _grid_ticks(cols, step)
    y_ticks = _grid_ticks(rows, step)
    ax.set_xticks(x_ticks + 0.5)
    ax.set_yticks(y_ticks + 0.5)
    ax.set_xticklabels(x_ticks, fontname="Times New Roman", fontsize=10)
    ax.set_yticklabels(y_ticks, fontname="Times New Roman", fontsize=10)
    ax.tick_params(axis="both", length=0)


def _hide_axis_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)


def _is_ship_map(map_file: str | Path | None) -> bool:
    if map_file is None:
        return False
    return Path(map_file).name == SHIP_MAP_NAME


def _resolve_ship_image_path(map_file: str | Path | None) -> Path | None:
    candidates = []
    if map_file is not None:
        candidates.append(Path(map_file).with_name(SHIP_IMAGE_NAME))
    candidates.append(Path(__file__).resolve().with_name(SHIP_IMAGE_NAME))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _figure_size(grid_shape: tuple[int, int], ship_style: bool = False) -> tuple[float, float]:
    rows, cols = grid_shape
    if ship_style:
        aspect = cols / max(1, rows)
        width = 14.0
        return width, max(3.2, width / max(aspect, 0.1))
    return 7.0, 7.0


def _draw_grid_map(ax, grid_map: np.ndarray) -> None:
    import matplotlib.patches as patches

    rows, cols = grid_map.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for row, col in np.argwhere(grid_map == 1):
        ax.add_patch(
            patches.Rectangle(
                (float(col), float(row)),
                1,
                1,
                facecolor=OBSTACLE_COLOR,
                edgecolor="none",
                zorder=0,
            )
        )


def _draw_ship_grid_map(ax, grid_map: np.ndarray, map_file: str | Path | None) -> None:
    import matplotlib.patches as patches

    rows, cols = grid_map.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_facecolor(SHIP_OBSTACLE_COLOR)

    for row, col in np.argwhere(grid_map == 0):
        ax.add_patch(
            patches.Rectangle(
                (float(col), float(row)),
                1,
                1,
                facecolor=SHIP_FREE_COLOR,
                edgecolor="none",
                zorder=0,
            )
        )

    image_path = _resolve_ship_image_path(map_file)
    if image_path is None:
        return

    import matplotlib.image as mpimg

    image = mpimg.imread(image_path)
    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=-1)
    else:
        rgb = image[..., :3]
    if rgb.dtype.kind in {"u", "i"}:
        rgb = rgb.astype(np.float32) / 255.0

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    line_mask = gray <= (230.0 / 255.0)
    rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = line_mask.astype(np.float32)

    height, width = gray.shape
    left = (0 - SHIP_TX1) / SHIP_XYDAIM - SHIP_XMIN
    right = (width - SHIP_TX1) / SHIP_XYDAIM - SHIP_XMIN
    top = (0 - SHIP_TY1) / SHIP_XYDAIM - SHIP_YMIN
    bottom = (height - SHIP_TY1) / SHIP_XYDAIM - SHIP_YMIN
    ax.imshow(
        rgba,
        extent=(left, right, bottom, top),
        origin="upper",
        interpolation="nearest",
        zorder=4,
    )


def _draw_grid_lines(ax, grid_shape: tuple[int, int], color: str = GRID_LINE_COLOR) -> None:
    rows, cols = grid_shape
    ax.vlines(np.arange(0, cols + 1), 0, rows, colors=color, linewidth=0.5, zorder=5)
    ax.hlines(np.arange(0, rows + 1), 0, cols, colors=color, linewidth=0.5, zorder=5)


def _draw_cell(ax, point, color, zorder: int = 3) -> None:
    import matplotlib.patches as patches

    row, col = int(point[0]), int(point[1])
    ax.add_patch(
        patches.Rectangle(
            (col, row),
            1,
            1,
            facecolor=color,
            edgecolor="none",
            zorder=zorder,
        )
    )


def _draw_start_circle(ax, point, color, zorder: int = 7) -> None:
    import matplotlib.patches as patches

    row, col = int(point[0]), int(point[1])
    ax.add_patch(
        patches.Circle(
            (col + 0.5, row + 0.5),
            radius=0.34,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=zorder,
        )
    )


def _as_path_array(path) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :2]


def _trim_after_first_exit(arr: np.ndarray, exits) -> np.ndarray:
    if len(arr) == 0:
        return arr
    exit_points = {tuple(int(v) for v in point) for point in np.asarray(exits, dtype=np.int32).reshape(-1, 2)}
    for idx, point in enumerate(arr):
        if tuple(int(v) for v in point[:2]) in exit_points:
            return arr[: idx + 1]
    return arr


def _draw_static_cells(ax, paths: Sequence[Sequence[np.ndarray]], exits) -> None:
    seen_starts: set[tuple[int, int]] = set()
    for agent_id, path in enumerate(paths):
        arr = _as_path_array(path)
        if len(arr) == 0:
            continue
        start = tuple(int(x) for x in arr[0])
        if start not in seen_starts:
            _draw_start_circle(ax, start, _agent_color(agent_id))
            seen_starts.add(start)

    for exit_point in exits:
        _draw_cell(ax, exit_point, EXIT_COLOR, zorder=6)


def _draw_map_base(
    ax,
    grid_map: np.ndarray,
    paths: Sequence[Sequence[np.ndarray]] = (),
    exits=(),
    map_file: str | Path | None = None,
) -> None:
    if _is_ship_map(map_file):
        _draw_ship_grid_map(ax, grid_map, map_file)
        grid_color = SHIP_GRID_LINE_COLOR
    else:
        _draw_grid_map(ax, grid_map)
        grid_color = GRID_LINE_COLOR
    _draw_static_cells(ax, paths, exits)
    if _is_ship_map(map_file):
        _hide_axis_ticks(ax)
    else:
        _set_grid_axis_ticks(ax, grid_map.shape)
    _draw_grid_lines(ax, grid_map.shape, color=grid_color)


def plot_paths(
    grid_map: np.ndarray,
    paths: Sequence[Sequence[np.ndarray]],
    exits,
    save_path: str | Path,
    map_file: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=_figure_size(grid_map.shape, _is_ship_map(map_file)))
    _draw_map_base(ax, grid_map, paths, exits, map_file=map_file)

    for i, path in enumerate(paths):
        arr = _as_path_array(path)
        if len(arr) == 0:
            continue
        ax.plot(
            arr[:, 1] + 0.5,
            arr[:, 0] + 0.5,
            color=_agent_color(i),
            linewidth=2,
            label=f"agent {i + 1}",
            zorder=4,
        )

    if len(paths) <= 12:
        ax.legend(loc="upper right", prop={"family": "Times New Roman", "size": 14})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_path_heatmap(
    grid_map: np.ndarray,
    paths: Sequence[Sequence[np.ndarray]],
    exits,
    save_path: str | Path,
    map_file: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heat = np.zeros(grid_map.shape, dtype=np.float32)
    rows, cols = grid_map.shape
    for path in paths:
        arr = _trim_after_first_exit(_as_path_array(path), exits).astype(np.int32)
        for row, col in arr:
            if 0 <= row < rows and 0 <= col < cols:
                heat[row, col] += 1.0

    fig, ax = plt.subplots(figsize=_figure_size(grid_map.shape, _is_ship_map(map_file)))
    _draw_map_base(ax, grid_map, paths, exits, map_file=map_file)

    masked_heat = np.ma.masked_where(heat <= 0, heat)
    if np.any(heat > 0):
        mesh = ax.pcolormesh(
            np.arange(cols + 1),
            np.arange(rows + 1),
            masked_heat,
            cmap=_congestion_heatmap_cmap(),
            shading="flat",
            alpha=0.65,
            zorder=2,
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("stay time steps")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def animate_path_heatmap(
    grid_map: np.ndarray,
    paths: Sequence[Sequence[np.ndarray]],
    exits,
    save_path: str | Path,
    fps: int = 8,
    map_file: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trimmed_paths = [_trim_after_first_exit(_as_path_array(path), exits).astype(np.int32) for path in paths]
    trimmed_paths = [arr for arr in trimmed_paths if len(arr) > 0]
    if not trimmed_paths:
        trimmed_paths = [np.zeros((1, 2), dtype=np.int32)]

    rows, cols = grid_map.shape
    n_frames = max(len(arr) for arr in trimmed_paths)
    final_heat = np.zeros(grid_map.shape, dtype=np.float32)
    for arr in trimmed_paths:
        for row, col in arr:
            if 0 <= row < rows and 0 <= col < cols:
                final_heat[row, col] += 1.0
    vmax = max(1.0, float(final_heat.max()))

    fig, ax = plt.subplots(figsize=_figure_size(grid_map.shape, _is_ship_map(map_file)))
    _draw_map_base(ax, grid_map, trimmed_paths, exits, map_file=map_file)
    heat = np.zeros(grid_map.shape, dtype=np.float32)
    masked_heat = np.ma.masked_where(heat <= 0, heat)
    mesh = ax.pcolormesh(
        np.arange(cols + 1),
        np.arange(rows + 1),
        masked_heat,
        cmap=_congestion_heatmap_cmap(),
        shading="flat",
        alpha=0.65,
        vmin=0,
        vmax=vmax,
        zorder=2,
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("cumulative stay time steps")

    def update(frame: int):
        heat.fill(0.0)
        for arr in trimmed_paths:
            end = min(frame + 1, len(arr))
            for row, col in arr[:end]:
                if 0 <= row < rows and 0 <= col < cols:
                    heat[row, col] += 1.0
        masked = np.ma.masked_where(heat <= 0, heat)
        mesh.set_array(masked.ravel())
        ax.set_title(f"Congestion heatmap step {frame}/{n_frames - 1}")
        return [mesh]

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


def plot_training_curves(csv_path: str | Path, save_path: str | Path, x_axis: str | None = None) -> None:
    import csv
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    if x_axis == "episode" and "episode" in rows[0]:
        x_values = [int(float(row["episode"])) for row in rows]
        x_label = "episode"
    elif x_axis == "step" and "global_step" in rows[0]:
        x_values = [int(float(row["global_step"])) for row in rows]
        x_label = "global step"
    elif x_axis == "update" and "update" in rows[0]:
        x_values = [int(float(row["update"])) for row in rows]
        x_label = "update"
    elif "global_step" in rows[0]:
        x_values = [int(float(row["global_step"])) for row in rows]
        x_label = "global step"
    elif "update" in rows[0]:
        x_values = [int(float(row["update"])) for row in rows]
        x_label = "update"
    else:
        x_values = [int(float(row["episode"])) for row in rows]
        x_label = "episode"

    reward_key = "recent_reward" if "recent_reward" in rows[0] else "episode_reward"
    evacuated_key = "recent_evacuated" if "recent_evacuated" in rows[0] else "num_evacuated"
    steps_key = "recent_episode_steps" if "recent_episode_steps" in rows[0] else "episode_steps"
    rewards = [float(row.get(reward_key, 0.0) or 0.0) for row in rows]
    evacuated = [float(row.get(evacuated_key, 0.0) or 0.0) for row in rows]
    steps = [float(row.get(steps_key, 0.0) or 0.0) for row in rows]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(x_values, rewards)
    axes[0].set_ylabel("reward")
    axes[1].plot(x_values, evacuated)
    axes[1].set_ylabel("evacuated")
    axes[2].plot(x_values, steps)
    axes[2].set_ylabel("steps")
    axes[2].set_xlabel(x_label)
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
    map_file: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    padded_paths = _pad_paths(paths)
    n_agents, n_frames, _ = padded_paths.shape

    if _is_ship_map(map_file):
        fig_size = _figure_size(grid_map.shape, ship_style=True)
    else:
        aspect = grid_map.shape[1] / max(1, grid_map.shape[0])
        fig_width = min(12.0, max(6.0, 7.0 * aspect))
        fig_height = min(12.0, max(6.0, 7.0 / max(aspect, 0.4)))
        fig_size = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=fig_size)
    _draw_map_base(ax, grid_map, paths, exits, map_file=map_file)

    lines = []
    markers = []
    labels = []
    for i in range(n_agents):
        color = _agent_color(i)
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.85, zorder=6)
        marker = ax.scatter([], [], color=color, marker="o", s=45, edgecolor="black", linewidth=0.4, zorder=7)
        lines.append(line)
        markers.append(marker)
        if n_agents <= 20:
            labels.append(ax.text(0, 0, str(i + 1), color="black", fontsize=8, ha="center", va="center", zorder=8))

    def update(frame: int):
        artists = []
        for i in range(n_agents):
            segment = padded_paths[i, : frame + 1]
            current = segment[-1]
            if trail:
                lines[i].set_data(segment[:, 1] + 0.5, segment[:, 0] + 0.5)
            else:
                lines[i].set_data([], [])
            markers[i].set_offsets([[current[1] + 0.5, current[0] + 0.5]])
            artists.extend([lines[i], markers[i]])
            if labels:
                labels[i].set_position((current[1] + 0.5, current[0] + 0.5))
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
