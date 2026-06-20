from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mappo import MAPPOAgent, MAPPOConfig, RolloutBuffer
from mappo_env import EvacuationConfig, MultiAgentGridEvacuationEnv
from train import (
    _points_to_arg,
    agent_arrival_record_summary,
    build_algorithm_save_dir,
    init_agent_arrival_records,
    make_update_training_log_row,
    reset_agent_arrival_record,
    save_initial_map,
    update_agent_arrival_records,
)
from utils import find_border_exits, find_marked_exits, load_txt_map, normalize_map_markers, parse_points, choose_random_starts, write_csv
from vec_env import make_vec_env
from visualize import animate_path_heatmap, animate_paths, plot_path_heatmap, plot_paths, plot_training_curves


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Map.txt experiment settings
# =============================================================================
# This file is the code-config version of training.
# In normal use, edit the settings below and run only:
#
#     python train_MAPPO.py
#
# You do not need to type map, exit, start, or agent information in the terminal.
#
# Coordinate format is always (row, col), starting from 0.
# For your current map.txt, rows are 0-103 and columns are 0-437.
#
# 1) Map file:
#    By default, train_MAPPO.py uses MAPPO_Multiagent_Evacuation/map.txt.
#
# 2) Exit locations:
#    Fill in all internal exits here. Every exit cell must be 0 in map.txt.
#    Example: EXITS = [(30, 120), (45, 210), (70, 350)]
#
# 3) Number of agents:
#    NUM_AGENTS controls how many evacuees are created.
#
# 4) Agent starting locations:
#    If STARTS is empty, the code automatically chooses NUM_AGENTS free cells
#    far away from the exits. If you want exact initial positions, fill STARTS
#    with exactly NUM_AGENTS coordinates.
#    Example: STARTS = [(90, 50), (91, 50), (92, 50), (93, 50)]
# =============================================================================

USE_CODE_CONFIG = True

MAP_FILE = Path(__file__).resolve().parent / "迷宫地图.txt"

# Fill all internal exits here. Every exit must be a 0 cell in map.txt.
# Example:
#EXITS = [(30, 120), (45, 210), (70, 350)]
# 如果终点为空，也就是不手动传 --exits，并且 train_MAPPO.py 里的 EXITS = []，程序会自动读取地图 txt 中数值为 2 的格子作为终点
EXITS: list[tuple[int, int]] = []
# 如果起点为空，也就是不手动传 --starts，并且 train_MAPPO.py 里的 STARTS = []，程序会根据 NUM_AGENTS / --num-agents 在可通行区域随机选择不重复的智能体起点
# Number of evacuees/agents.
NUM_AGENTS = 30

# Fill exact starts here if you want fixed initial positions.
# If STARTS is empty, train_MAPPO.py automatically chooses NUM_AGENTS free cells far from exits.
# If STARTS is not empty, its length must equal NUM_AGENTS.
# Example:
#STARTS = [(90, 50), (91, 50), (92, 50), (93, 50), (94, 50), (95, 50), (96, 50), (97, 50)]

STARTS: list[tuple[int, int]] = [(46, 22), (21, 42), (18, 42), (5, 19), (18, 18), (36, 9), (8, 27), (32, 44), (19, 10), (40, 17), (41, 19), (4, 16), (3, 39), (47, 3), (37, 19), (46, 33), (22, 12), (6, 35), (32, 38), (11, 25), (1, 36), (26, 14), (17, 5), (26, 47), (42, 41), (42, 46), (11, 21), (16, 24), (9, 12), (24, 1)]

SEED = 1
RUN_SEEDS = [1, 2, 3, 4, 5]

MAX_STEPS = 500
#TOTAL_TIMESTEPS = 500_000
TOTAL_TIMESTEPS = 204800
ROLLOUT_STEPS = 128
UPDATE_EPOCHS = 8
MINIBATCH_SIZE = 512
NUM_ENVS = 8
VEC_ENV_TYPE = "subproc"
LOCAL_VIEW_RADIUS = 5
MAX_NEIGHBORS = 4
EXIT_CAPACITY = 1
NO_WAIT = False
SAVE_ANIMATION = False
ANIMATION_FORMAT = "mp4"
ANIMATION_FPS = 8
SAVE_DIR = None


def parse_points_file(path: str | Path | None) -> list[tuple[int, int]]:
    if not path:
        return []
    points: list[tuple[int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row, col = line.replace(",", " ").split()[:2]
        points.append((int(row), int(col)))
    return points


def validate_points(grid: np.ndarray, points: list[tuple[int, int]], name: str) -> None:
    rows, cols = grid.shape
    if len(set(points)) != len(points):
        raise ValueError(f"{name} contains duplicate coordinates: {points}")
    for row, col in points:
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"{name} point {(row, col)} is outside map shape {rows}x{cols}.")
        if grid[row, col] != 0:
            raise ValueError(f"{name} point {(row, col)} is not walkable. It must be 0 in map.txt.")


def save_episode_outputs(
    save_dir: Path,
    prefix: str,
    env: MultiAgentGridEvacuationEnv,
    paths,
    summary: dict,
    save_animation: bool,
    animation_format: str,
    fps: int,
    map_file: str | Path | None = None,
) -> None:
    plot_paths(env.grid_map, paths, env.exits, save_dir / f"{prefix}_paths.png", map_file=map_file)
    plot_path_heatmap(env.grid_map, paths, env.exits, save_dir / f"{prefix}_heatmap.png", map_file=map_file)

    path_dir = save_dir / f"{prefix}_paths"
    path_dir.mkdir(parents=True, exist_ok=True)
    npz_data = {}
    for agent_id, path in enumerate(paths):
        arr = np.asarray(path, dtype=np.int32)
        np.savetxt(path_dir / f"agent_{agent_id:02d}.txt", arr, fmt="%d")
        npz_data[f"agent_{agent_id:02d}"] = arr
    if npz_data:
        np.savez_compressed(save_dir / f"{prefix}_paths.npz", **npz_data)

    with (save_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_animation and paths:
        animation_path = animate_paths(
            env.grid_map,
            paths,
            env.exits,
            save_dir / f"{prefix}_episode.{animation_format}",
            fps=fps,
            map_file=map_file,
        )
        print(f"{prefix.capitalize()} episode animation saved to: {animation_path}")
        heatmap_animation_path = animate_path_heatmap(
            env.grid_map,
            paths,
            env.exits,
            save_dir / f"{prefix}_heatmap_episode.{animation_format}",
            fps=fps,
            map_file=map_file,
        )
        print(f"{prefix.capitalize()} heatmap animation saved to: {heatmap_animation_path}")


def _path_array(path) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :2]


def _trim_path_at_first_exit(arr: np.ndarray, exit_points: set[tuple[int, int]]) -> np.ndarray:
    trim_end = len(arr)
    for step_idx, point in enumerate(arr):
        if tuple(int(v) for v in point[:2]) in exit_points:
            trim_end = step_idx + 1
            break
    return arr[:trim_end]


def compute_agent_congestion(paths, exits, num_agents: int) -> tuple[list[int], list[int]]:
    exit_points = {tuple(int(v) for v in point) for point in np.asarray(exits, dtype=np.int32).tolist()}
    congestion_counts: list[int] = []
    congestion_durations: list[int] = []
    for agent_id in range(int(num_agents)):
        arr = _path_array(paths[agent_id] if paths is not None and agent_id < len(paths) else [])
        trimmed = _trim_path_at_first_exit(arr, exit_points)
        count = 0
        duration = 0
        in_congestion = False
        for step_idx in range(1, len(trimmed)):
            stopped = np.array_equal(trimmed[step_idx], trimmed[step_idx - 1])
            if stopped:
                duration += 1
                if not in_congestion:
                    count += 1
                    in_congestion = True
            else:
                in_congestion = False
        congestion_counts.append(count)
        congestion_durations.append(duration)
    return congestion_counts, congestion_durations


def compute_episode_metrics(paths, exits, num_agents: int, summary: dict | None = None) -> dict:
    summary = summary or {}
    exit_points = [tuple(int(v) for v in point) for point in np.asarray(exits, dtype=np.int32).tolist()]
    counts = {point: 0 for point in exit_points}
    total_agents = int(num_agents)
    raw_arrival_steps = list(summary.get("arrival_steps", []))
    raw_collision_counts = list(summary.get("collision_counts", []))
    raw_arrival_time_seconds = list(summary.get("arrival_time_seconds", []))
    raw_exit_positions = list(summary.get("exit_positions", []))
    arrival_steps: list[int] = []
    arrival_time_seconds: list[float] = []
    exit_positions: list[list[int]] = []
    path_lengths: list[float] = []
    collision_counts: list[int] = []
    congestion_counts, congestion_durations = compute_agent_congestion(paths, exits, total_agents)

    for agent_id in range(total_agents):
        arr = _path_array(paths[agent_id] if paths is not None and agent_id < len(paths) else [])
        first_exit_step = -1
        first_exit_position = [-1, -1]
        trim_end = len(arr)
        for step_idx, point in enumerate(arr):
            point_key = tuple(int(v) for v in point[:2])
            if point_key in counts:
                counts[point_key] += 1
                first_exit_step = step_idx
                first_exit_position = [int(point_key[0]), int(point_key[1])]
                trim_end = step_idx + 1
                break

        if agent_id < len(raw_arrival_steps) and int(raw_arrival_steps[agent_id]) >= 0:
            arrival_step = int(raw_arrival_steps[agent_id])
        else:
            arrival_step = first_exit_step
        arrival_steps.append(arrival_step)

        if agent_id < len(raw_arrival_time_seconds) and float(raw_arrival_time_seconds[agent_id]) >= 0.0:
            arrival_time_seconds.append(float(raw_arrival_time_seconds[agent_id]))
        else:
            arrival_time_seconds.append(-1.0)

        exit_position = first_exit_position
        if agent_id < len(raw_exit_positions):
            raw_pos = list(raw_exit_positions[agent_id])
            if len(raw_pos) >= 2 and int(raw_pos[0]) >= 0 and int(raw_pos[1]) >= 0:
                exit_position = [int(raw_pos[0]), int(raw_pos[1])]
        exit_positions.append(exit_position)

        trimmed = arr[:trim_end]
        if len(trimmed) <= 1:
            path_lengths.append(0.0)
        else:
            diffs = np.diff(trimmed, axis=0)
            path_lengths.append(float(np.sum(np.linalg.norm(diffs, axis=1))))

        if agent_id < len(raw_collision_counts):
            collision_counts.append(int(raw_collision_counts[agent_id]))
        else:
            collision_counts.append(0)

    evacuated = int(sum(counts.values()))
    successful_arrivals = [step for step in arrival_steps if step >= 0]
    success_rate = evacuated / max(1, total_agents)
    all_evacuated = evacuated == total_agents
    total_evacuation_time = max(successful_arrivals) if all_evacuated and successful_arrivals else -1
    average_evacuation_time = float(np.mean(successful_arrivals)) if successful_arrivals else -1.0
    exits_usage = []
    for point in exit_points:
        count = int(counts[point])
        exits_usage.append(
            {
                "exit": list(point),
                "count": count,
                "rate_by_agents": count / max(1, total_agents),
                "rate_by_evacuated": count / evacuated if evacuated > 0 else 0.0,
            }
        )
    return {
        "total_agents": total_agents,
        "evacuated_agents": evacuated,
        "success_rate": success_rate,
        "total_evacuation_time": int(total_evacuation_time),
        "average_evacuation_time": average_evacuation_time,
        "arrival_steps": arrival_steps,
        "arrival_time_seconds": arrival_time_seconds,
        "exit_positions": exit_positions,
        "path_lengths": path_lengths,
        "average_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "collision_counts": collision_counts,
        "total_collision_count": int(sum(collision_counts)),
        "average_collision_count": float(np.mean(collision_counts)) if collision_counts else 0.0,
        "congestion_counts": congestion_counts,
        "congestion_durations": congestion_durations,
        "total_congestion_count": int(sum(congestion_counts)),
        "average_congestion_count": float(np.mean(congestion_counts)) if congestion_counts else 0.0,
        "total_congestion_duration": int(sum(congestion_durations)),
        "average_congestion_duration": float(np.mean(congestion_durations)) if congestion_durations else 0.0,
        "exits": exits_usage,
    }


def _append_episode_metrics(lines: list[str], title: str, metrics: dict | None) -> None:
    lines.append(f"{title}:")
    if metrics is None:
        lines.append("  not_available")
        return
    lines.append(f"  evacuated_agents: {metrics['evacuated_agents']} / {metrics['total_agents']}")
    lines.append(f"  evacuation_success_rate: {metrics['success_rate']:.6f}")
    lines.append(f"  total_evacuation_time: {metrics['total_evacuation_time']}")
    lines.append(f"  average_evacuation_time: {metrics['average_evacuation_time']:.6f}")
    lines.append(f"  average_path_length: {metrics['average_path_length']:.6f}")
    lines.append(f"  total_collision_count: {metrics['total_collision_count']}")
    lines.append(f"  average_collision_count: {metrics['average_collision_count']:.6f}")
    lines.append(f"  total_congestion_count: {metrics['total_congestion_count']}")
    lines.append(f"  average_congestion_count: {metrics['average_congestion_count']:.6f}")
    lines.append(f"  total_congestion_duration_steps: {metrics['total_congestion_duration']}")
    lines.append(f"  average_congestion_duration_steps: {metrics['average_congestion_duration']:.6f}")
    lines.append("  exit_usage:")
    for idx, item in enumerate(metrics["exits"]):
        row, col = item["exit"]
        lines.append(
            f"    exit_{idx} ({row}, {col}): "
            f"count={item['count']}, "
            f"rate_by_agents={item['rate_by_agents']:.6f}, "
            f"rate_by_evacuated={item['rate_by_evacuated']:.6f}"
        )


def save_agent_metrics_csv(
    save_dir: Path,
    last_metrics: dict | None = None,
    best_metrics: dict | None = None,
) -> Path | None:
    rows = []
    for episode_name, metrics in (("last", last_metrics), ("best", best_metrics)):
        if metrics is None:
            continue
        arrivals = list(metrics.get("arrival_steps", []))
        arrival_time_seconds = list(metrics.get("arrival_time_seconds", []))
        exit_positions = list(metrics.get("exit_positions", []))
        path_lengths = list(metrics.get("path_lengths", []))
        collisions = list(metrics.get("collision_counts", []))
        counts = list(metrics.get("congestion_counts", []))
        durations = list(metrics.get("congestion_durations", []))
        n_agents = max(
            len(arrivals),
            len(arrival_time_seconds),
            len(exit_positions),
            len(path_lengths),
            len(collisions),
            len(counts),
            len(durations),
        )
        for agent_idx in range(n_agents):
            exit_position = exit_positions[agent_idx] if agent_idx < len(exit_positions) else [-1, -1]
            if len(exit_position) < 2:
                exit_position = [-1, -1]
            rows.append(
                {
                    "episode": episode_name,
                    "agent": f"agent {agent_idx + 1}",
                    "agent_id": agent_idx + 1,
                    "arrival_step": arrivals[agent_idx] if agent_idx < len(arrivals) else -1,
                    "arrival_time_seconds": (
                        arrival_time_seconds[agent_idx] if agent_idx < len(arrival_time_seconds) else -1.0
                    ),
                    "exit_row": int(exit_position[0]),
                    "exit_col": int(exit_position[1]),
                    "path_length": path_lengths[agent_idx] if agent_idx < len(path_lengths) else 0.0,
                    "collision_count": collisions[agent_idx] if agent_idx < len(collisions) else 0,
                    "congestion_count": counts[agent_idx] if agent_idx < len(counts) else 0,
                    "congestion_duration_steps": durations[agent_idx] if agent_idx < len(durations) else 0,
                }
            )
    if not rows:
        return None
    csv_path = save_dir / "agent_metrics.csv"
    write_csv(csv_path, rows)
    return csv_path


def save_run_info_txt(
    save_dir: Path,
    args,
    env: MultiAgentGridEvacuationEnv,
    last_metrics: dict | None = None,
    best_metrics: dict | None = None,
    training_started_at: dt.datetime | None = None,
    training_finished_at: dt.datetime | None = None,
) -> Path:
    exits = [tuple(int(v) for v in point) for point in env.exits.tolist()]
    starts = [tuple(int(v) for v in point) for point in env.starts.tolist()]
    training_duration_seconds = None
    if training_started_at is not None and training_finished_at is not None:
        training_duration_seconds = (training_finished_at - training_started_at).total_seconds()
    lines = [
        "run_info",
        f"seed: {args.seed}",
        f"map_file: {Path(args.map_file).resolve()}",
        f"map_shape: {env.rows} x {env.cols}",
        f"num_agents: {env.n_agents}",
    ]
    if training_started_at is not None:
        lines.append(f"training_started_at: {training_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if training_finished_at is not None:
        lines.append(f"training_finished_at: {training_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if training_duration_seconds is not None:
        lines.append(f"training_duration_seconds: {training_duration_seconds:.3f}")
        lines.append(f"training_duration_hms: {dt.timedelta(seconds=round(training_duration_seconds))}")
    lines.append("exits:")
    lines.extend(f"  exit_{idx}: ({row}, {col})" for idx, (row, col) in enumerate(exits))
    lines.append("starts:")
    lines.extend(f"  agent_{idx}: ({row}, {col})" for idx, (row, col) in enumerate(starts))
    _append_episode_metrics(lines, "last_episode_metrics", last_metrics)
    _append_episode_metrics(lines, "best_episode_metrics", best_metrics)
    info_path = save_dir / "run_info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info_path


def build_map_txt_env(args) -> MultiAgentGridEvacuationEnv:
    raw_grid = load_txt_map(args.map_file)
    exits = parse_points(args.exits) or parse_points_file(args.exits_file) or find_marked_exits(raw_grid) or list(EXITS)
    grid = normalize_map_markers(raw_grid)
    exits = exits or find_border_exits(grid)
    if not exits:
        raise ValueError(
            "No exits were configured. Set EXITS in train_MAPPO.py or mark exits as 2 in the map txt."
        )
    validate_points(grid, exits, "Exit")

    starts = parse_points(args.starts) or parse_points_file(args.starts_file) or list(STARTS)
    if starts:
        if len(starts) != args.num_agents:
            raise ValueError(
                f"STARTS has {len(starts)} points, but num_agents is {args.num_agents}. "
                "Please make them equal."
        )
        validate_points(grid, starts, "Start")
    else:
        starts = choose_random_starts(grid, exits, args.num_agents)

    cfg = EvacuationConfig(
        max_steps=args.max_steps,
        local_view_radius=args.local_view_radius,
        max_neighbors=args.max_neighbors,
        allow_wait=not args.no_wait,
        exit_capacity=args.exit_capacity,
    )
    return MultiAgentGridEvacuationEnv(grid, starts, exits, cfg)


def freeze_shared_starts(args, env_builder=build_map_txt_env):
    probe_env = env_builder(args)
    starts = [tuple(int(v) for v in point) for point in probe_env.starts.tolist()]
    args.starts = _points_to_arg(starts)
    args.starts_file = None
    return args


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MAPPO on MAPPO_Multiagent_Evacuation/map.txt with explicit internal exits."
    )
    parser.add_argument("--map-file", type=str, default=str(MAP_FILE), help="Default: ./map.txt")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--starts", type=str, default=None, help='Format: "r,c;r,c;..."')
    parser.add_argument("--starts-file", type=str, default=None, help="One start point per line: row,col")
    parser.add_argument("--exits", type=str, default=None, help='Format: "r,c;r,c;..."')
    parser.add_argument("--exits-file", type=str, default=None, help="One exit point per line: row,col")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--local-view-radius", type=int, default=LOCAL_VIEW_RADIUS)
    parser.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    parser.add_argument("--exit-capacity", type=int, default=EXIT_CAPACITY)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
    parser.add_argument("--minibatch-size", type=int, default=MINIBATCH_SIZE)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Number of synchronous vectorized environments.")
    parser.add_argument("--vec-env-type", choices=["dummy", "subproc"], default=VEC_ENV_TYPE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-animation", action="store_true", default=not SAVE_ANIMATION)
    parser.add_argument("--animation-format", choices=["mp4", "gif"], default=ANIMATION_FORMAT)
    parser.add_argument("--animation-fps", type=int, default=ANIMATION_FPS)
    return parser


def make_code_config_args() -> SimpleNamespace:
    return SimpleNamespace(
        map_file=str(MAP_FILE),
        num_agents=NUM_AGENTS,
        starts=None,
        starts_file=None,
        exits=None,
        exits_file=None,
        max_steps=MAX_STEPS,
        local_view_radius=LOCAL_VIEW_RADIUS,
        max_neighbors=MAX_NEIGHBORS,
        exit_capacity=EXIT_CAPACITY,
        no_wait=NO_WAIT,
        total_timesteps=TOTAL_TIMESTEPS,
        rollout_steps=ROLLOUT_STEPS,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        num_envs=NUM_ENVS,
        vec_env_type=VEC_ENV_TYPE,
        seed=SEED,
        save_dir=SAVE_DIR,
        no_animation=not SAVE_ANIMATION,
        animation_format=ANIMATION_FORMAT,
        animation_fps=ANIMATION_FPS,
    )


def main(seed_override: int | None = None) -> None:
    args = make_code_config_args() if USE_CODE_CONFIG else make_parser().parse_args()
    if seed_override is not None:
        args.seed = int(seed_override)
    training_started_at = dt.datetime.now()
    if args.num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    set_random_seed(args.seed)

    args = freeze_shared_starts(args, build_map_txt_env)
    vec_env = make_vec_env([partial(build_map_txt_env, args) for _ in range(args.num_envs)], args.vec_env_type)
    env = vec_env.envs[0]
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = build_algorithm_save_dir(args, "MAPPO", run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = MAPPOConfig(
        
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    agent = MAPPOAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)

    print(f"Use code config: {USE_CODE_CONFIG}")
    print(f"Device: {cfg.device}")
    print(f"Map file: {Path(args.map_file).resolve()}")
    print(f"Map: {env.rows}x{env.cols}")
    print(f"Agents: {env.n_agents}")
    print(f"Exits: {[tuple(x) for x in env.exits.tolist()]}")
    print(f"Starts: {[tuple(x) for x in env.starts.tolist()]}")
    print(f"Vectorized envs: {vec_env.num_envs}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Save dir: {save_dir}")

    experiment_config = {
        "map_file": str(Path(args.map_file).resolve()),
        "map_shape": [env.rows, env.cols],
        "num_agents": env.n_agents,
        "exits": [list(x) for x in env.exits.tolist()],
        "starts": [list(x) for x in env.starts.tolist()],
        "max_steps": args.max_steps,
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "num_envs": args.num_envs,
        "vec_env_type": args.vec_env_type,
        "local_view_radius": args.local_view_radius,
        "max_neighbors": args.max_neighbors,
        "exit_capacity": args.exit_capacity,
        "allow_wait": not args.no_wait,
        "save_animation": not args.no_animation,
        "animation_format": args.animation_format,
        "seed": args.seed,
    }
    with (save_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    info_path = save_run_info_txt(save_dir, args, env, training_started_at=training_started_at)
    print(f"Run info: {info_path}")
    initial_map_path = save_initial_map(save_dir, env, args.map_file)
    print(f"Initial map: {initial_map_path}")

    obs, infos = vec_env.reset(seed=args.seed)
    episode_reward = np.zeros(vec_env.num_envs, dtype=np.float32)
    episode_steps = np.zeros(vec_env.num_envs, dtype=np.int32)
    episode_collision_counts = np.zeros((vec_env.num_envs, env.n_agents), dtype=np.int32)
    episode_arrival_time_seconds, episode_exit_positions = init_agent_arrival_records(vec_env.num_envs, env.n_agents)
    episode_idx = 0
    episode_logs = []
    training_logs = []
    best_evacuated = -1
    best_steps = args.max_steps + 1
    best_path_steps = args.max_steps + 1
    last_episode_paths = None
    last_episode_summary = None
    best_episode_paths = None
    best_episode_summary = None
    fallback_episode_paths = None
    fallback_episode_summary = None

    num_updates = max(1, cfg.total_timesteps // (cfg.rollout_steps * vec_env.num_envs))
    global_step = 0
    last_done_tensor = torch.zeros((vec_env.num_envs, env.n_agents), dtype=torch.float32, device=cfg.device)
    for update in range(1, num_updates + 1):
        update_started_at = dt.datetime.now()
        buffer = RolloutBuffer(
            cfg.rollout_steps,
            env.n_agents,
            env.obs_dim,
            agent.global_obs_dim,
            cfg.device,
            num_envs=vec_env.num_envs,
        )
        for _ in range(cfg.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
            global_obs = obs_tensor.reshape(vec_env.num_envs, -1)
            flat_obs = obs_tensor.reshape(vec_env.num_envs * env.n_agents, env.obs_dim)
            active_mask_tensor = torch.tensor(
                np.stack([item["active"] for item in infos], axis=0),
                dtype=torch.float32,
                device=cfg.device,
            )
            available_actions_tensor = torch.tensor(
                np.stack(
                    [
                        item["available_actions"]
                        for item in infos
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
                device=cfg.device,
            )
            flat_available_actions = available_actions_tensor.reshape(
                vec_env.num_envs * env.n_agents,
                env.n_actions,
            )
            with torch.no_grad():
                actions_flat, logprobs_flat, values = agent.act(
                    flat_obs,
                    global_obs,
                    flat_available_actions,
                )
            actions = actions_flat.reshape(vec_env.num_envs, env.n_agents)
            logprobs = logprobs_flat.reshape(vec_env.num_envs, env.n_agents)

            next_obs, rewards, terminated, truncated, next_infos = vec_env.step(actions.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
            next_active_tensor = torch.tensor(
                np.stack([item["active"] for item in next_infos], axis=0),
                dtype=torch.float32,
                device=cfg.device,
            )
            done_tensor = 1.0 - next_active_tensor
            if np.any(done):
                done_tensor[torch.tensor(done, dtype=torch.bool, device=cfg.device)] = 1.0
            bad_mask_tensor = torch.ones((vec_env.num_envs, env.n_agents), dtype=torch.float32, device=cfg.device)
            time_limit = np.logical_and(truncated, np.logical_not(terminated))
            if np.any(time_limit):
                bad_mask_tensor[torch.tensor(time_limit, dtype=torch.bool, device=cfg.device)] = 0.0
            last_done_tensor = done_tensor
            buffer.add(
                obs_tensor,
                global_obs,
                actions,
                logprobs,
                reward_tensor,
                done_tensor,
                values,
                active_mask_tensor,
                available_actions_tensor,
                bad_mask_tensor,
            )

            episode_reward += rewards.sum(axis=1)
            episode_steps += 1
            global_step += vec_env.num_envs
            obs = next_obs
            infos = next_infos
            for info_idx, item in enumerate(infos):
                for agent_id, component in enumerate(item.get("reward_components", [])):
                    if component.get("collision", 0.0) < 0.0:
                        episode_collision_counts[info_idx, agent_id] += 1
            update_agent_arrival_records(
                episode_arrival_time_seconds,
                episode_exit_positions,
                infos,
                training_started_at,
            )

            for env_idx in np.where(done)[0]:
                done_info = infos[env_idx]
                num_evacuated = int(done_info["num_evacuated"])
                episode_summary = {
                    "episode": episode_idx,
                    "env_index": int(env_idx),
                    "global_step": global_step,
                    "episode_reward": round(float(episode_reward[env_idx]), 6),
                    "episode_steps": int(episode_steps[env_idx]),
                    "num_evacuated": num_evacuated,
                    "arrival_steps": done_info["arrival_steps"].tolist(),
                    "collision_counts": episode_collision_counts[env_idx].astype(int).tolist(),
                    "average_collision_count": round(float(np.mean(episode_collision_counts[env_idx])), 6),
                    **agent_arrival_record_summary(
                        episode_arrival_time_seconds,
                        episode_exit_positions,
                        int(env_idx),
                    ),
                }
                episode_logs.append(
                    {
                        "episode": episode_idx,
                        "env_index": int(env_idx),
                        "global_step": global_step,
                        "episode_reward": round(float(episode_reward[env_idx]), 6),
                        "episode_steps": int(episode_steps[env_idx]),
                        "num_evacuated": num_evacuated,
                    }
                )
                is_model_better = num_evacuated > best_evacuated or (
                    num_evacuated == best_evacuated and episode_steps[env_idx] < best_steps
                )
                if is_model_better:
                    best_evacuated = num_evacuated
                    best_steps = int(episode_steps[env_idx])
                    agent.save(str(save_dir / "best_model.pt"))
                episode_paths = done_info.get("paths")
                fully_evacuated = num_evacuated == env.n_agents and episode_paths is not None
                if fully_evacuated:
                    last_episode_paths = episode_paths
                    last_episode_summary = dict(episode_summary)
                    is_better = best_episode_paths is None or episode_steps[env_idx] < best_path_steps
                    if is_better:
                        best_path_steps = int(episode_steps[env_idx])
                        best_episode_paths = [[p.copy() for p in path] for path in episode_paths]
                        best_episode_summary = dict(episode_summary)
                elif best_episode_paths is None:
                    fallback_episode_paths = vec_env.get_paths(int(env_idx))
                    fallback_episode_summary = dict(episode_summary)

                episode_idx += 1
                episode_reward[env_idx] = 0.0
                episode_steps[env_idx] = 0
                episode_collision_counts[env_idx] = 0
                reset_agent_arrival_record(episode_arrival_time_seconds, episode_exit_positions, int(env_idx))
                reset_obs, reset_info = vec_env.reset_at(int(env_idx))
                obs[env_idx] = reset_obs
                infos[env_idx] = reset_info

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
        global_obs = obs_tensor.reshape(vec_env.num_envs, -1)
        with torch.no_grad():
            next_value = agent.critic(global_obs)
        buffer.compute_returns_and_advantages(
            next_value,
            last_done_tensor,
            cfg.gamma,
            cfg.gae_lambda,
            cfg.use_proper_time_limits,
            agent.value_normalizer,
        )
        metrics = agent.update(buffer)

        update_seconds = (dt.datetime.now() - update_started_at).total_seconds()
        training_row = make_update_training_log_row(
            update,
            num_updates,
            global_step,
            update_seconds,
            metrics,
            episode_logs,
        )
        training_logs.append(training_row)
        mean_reward = training_row["recent_reward"]
        mean_evac = training_row["recent_evacuated"]
        print(
            f"update={update:04d}/{num_updates} step={global_step:07d} "
            f"update_time={update_seconds:.2f}s "
            f"loss={metrics['loss']:.4f} entropy={metrics['entropy']:.4f} "
            f"recent_reward={mean_reward:.2f} recent_evacuated={mean_evac:.2f}"
        )

    agent.save(str(save_dir / "final_model.pt"))
    write_csv(save_dir / "episode_log.csv", episode_logs)
    write_csv(save_dir / "training_log.csv", training_logs)
    if training_logs:
        plot_training_curves(
            save_dir / "training_log.csv",
            save_dir / "training_curves_by_step.png",
            x_axis="step",
        )
        plot_training_curves(save_dir / "training_log.csv", save_dir / "training_curves.png", x_axis="step")
    if episode_logs:
        plot_training_curves(
            save_dir / "episode_log.csv",
            save_dir / "training_curves_by_episode.png",
            x_axis="episode",
        )
    if last_episode_paths is None:
        if fallback_episode_paths is not None:
            last_episode_paths = fallback_episode_paths
            last_episode_summary = fallback_episode_summary
        else:
            last_episode_paths = infos[0].get("paths") or vec_env.get_paths(0)
            last_episode_summary = {
                "episode": episode_idx,
                "global_step": global_step,
                "episode_reward": round(float(episode_reward[0]), 6),
                "episode_steps": int(episode_steps[0]),
                "num_evacuated": int(infos[0].get("num_evacuated", 0)),
                "arrival_steps": infos[0].get("arrival_steps", np.array([], dtype=np.int32)).tolist(),
                "collision_counts": episode_collision_counts[0].astype(int).tolist(),
                "average_collision_count": round(float(np.mean(episode_collision_counts[0])), 6),
                **agent_arrival_record_summary(episode_arrival_time_seconds, episode_exit_positions, 0),
            }

    if last_episode_paths:
        last_metrics = compute_episode_metrics(last_episode_paths, env.exits, env.n_agents, last_episode_summary or {})
        last_summary_to_save = dict(last_episode_summary or {})
        last_summary_to_save["metrics"] = last_metrics
        save_episode_outputs(
            save_dir,
            "last",
            env,
            last_episode_paths,
            last_summary_to_save,
            not args.no_animation,
            args.animation_format,
            args.animation_fps,
            args.map_file,
        )
    else:
        last_metrics = None
    if best_episode_paths:
        best_metrics = compute_episode_metrics(best_episode_paths, env.exits, env.n_agents, best_episode_summary or {})
        best_summary_to_save = dict(best_episode_summary or {})
        best_summary_to_save["metrics"] = best_metrics
        save_episode_outputs(
            save_dir,
            "best",
            env,
            best_episode_paths,
            best_summary_to_save,
            not args.no_animation,
            args.animation_format,
            args.animation_fps,
            args.map_file,
        )
    else:
        best_metrics = None
    training_finished_at = dt.datetime.now()
    save_agent_metrics_csv(save_dir, last_metrics, best_metrics)
    save_run_info_txt(
        save_dir,
        args,
        env,
        last_metrics,
        best_metrics,
        training_started_at=training_started_at,
        training_finished_at=training_finished_at,
    )
    vec_env.close()
    print("Training finished.")


def run_all_seeds() -> None:
    seeds = RUN_SEEDS if USE_CODE_CONFIG else [SEED]
    for seed in seeds:
        print(f"\n===== Running seed {seed} =====")
        main(seed_override=seed)


if __name__ == "__main__":
    run_all_seeds()
