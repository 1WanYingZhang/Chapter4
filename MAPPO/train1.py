from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mappo import MAPPOAgent, MAPPOConfig, RolloutBuffer
from mappo_env import EvacuationConfig, MultiAgentGridEvacuationEnv
from utils import choose_default_starts, load_txt_map, parse_points, write_csv
from vec_env import DummyVecEnv
from visualize import animate_paths, plot_paths, plot_training_curves


# =============================================================================
# Map.txt experiment settings
# =============================================================================
# This file is the code-config version of training.
# In normal use, edit the settings below and run only:
#
#     python train1.py
#
# You do not need to type map, exit, start, or agent information in the terminal.
#
# Coordinate format is always (row, col), starting from 0.
# For your current map.txt, rows are 0-103 and columns are 0-437.
#
# 1) Map file:
#    By default, train1.py uses MAPPO_Multiagent_Evacuation/map.txt.
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

MAP_FILE = Path(__file__).resolve().parent / "普通地图.txt"

# Fill all internal exits here. Every exit must be a 0 cell in map.txt.
# Example:
#EXITS = [(30, 120), (45, 210), (70, 350)]
#EXITS: list[tuple[int, int]] = []
EXITS = [(49, 20), (0, 45)]

# Number of evacuees/agents.
NUM_AGENTS = 3

# Fill exact starts here if you want fixed initial positions.
# If STARTS is empty, train1.py automatically chooses NUM_AGENTS free cells far from exits.
# If STARTS is not empty, its length must equal NUM_AGENTS.
# Example:
#STARTS = [(90, 50), (91, 50), (92, 50), (93, 50), (94, 50), (95, 50), (96, 50), (97, 50)]
#STARTS: list[tuple[int, int]] = []
STARTS = [(5, 30), (30, 5), (25, 43)]

MAX_STEPS = 1000
TOTAL_TIMESTEPS = 500_000
#TOTAL_TIMESTEPS = 51200
ROLLOUT_STEPS = 1024
UPDATE_EPOCHS = 8
MINIBATCH_SIZE = 512
NUM_ENVS = 1
LOCAL_VIEW_RADIUS = 5
MAX_NEIGHBORS = 4
EXIT_CAPACITY = 1
NO_WAIT = False
SAVE_ANIMATION = True
ANIMATION_FORMAT = "mp4"
ANIMATION_FPS = 8
SEED = 1
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
) -> None:
    plot_paths(env.grid_map, paths, env.exits, save_dir / f"{prefix}_paths.png")

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
        )
        print(f"{prefix.capitalize()} episode animation saved to: {animation_path}")


def build_map_txt_env(args) -> MultiAgentGridEvacuationEnv:
    grid = load_txt_map(args.map_file)
    exits = parse_points(args.exits) or parse_points_file(args.exits_file) or list(EXITS)
    if not exits:
        raise ValueError(
            "No exits were configured. Set EXITS in train1.py or run with "
            '--exits "row,col;row,col" or --exits-file exits.txt.'
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
        starts = choose_default_starts(grid, exits, args.num_agents)

    cfg = EvacuationConfig(
        max_steps=args.max_steps,
        local_view_radius=args.local_view_radius,
        max_neighbors=args.max_neighbors,
        allow_wait=not args.no_wait,
        exit_capacity=args.exit_capacity,
    )
    return MultiAgentGridEvacuationEnv(grid, starts, exits, cfg)


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
        seed=SEED,
        save_dir=SAVE_DIR,
        no_animation=not SAVE_ANIMATION,
        animation_format=ANIMATION_FORMAT,
        animation_fps=ANIMATION_FPS,
    )


def main() -> None:
    args = make_code_config_args() if USE_CODE_CONFIG else make_parser().parse_args()
    if args.num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vec_env = DummyVecEnv([lambda: build_map_txt_env(args) for _ in range(args.num_envs)])
    env = vec_env.envs[0]
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir or Path(__file__).resolve().parent / "results" / f"map_txt_train1_{run_id}")
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

    obs, infos = vec_env.reset(seed=args.seed)
    episode_reward = np.zeros(vec_env.num_envs, dtype=np.float32)
    episode_steps = np.zeros(vec_env.num_envs, dtype=np.int32)
    episode_idx = 0
    episode_logs = []
    best_evacuated = -1
    best_steps = args.max_steps + 1
    last_episode_paths = None
    last_episode_summary = None
    best_episode_paths = None
    best_episode_summary = None

    num_updates = max(1, cfg.total_timesteps // (cfg.rollout_steps * vec_env.num_envs))
    global_step = 0
    last_done_tensor = torch.zeros((vec_env.num_envs, env.n_agents), dtype=torch.float32, device=cfg.device)
    for update in range(1, num_updates + 1):
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
                        item.get("available_actions", vec_env.envs[idx].available_actions())
                        for idx, item in enumerate(infos)
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

            for env_idx in np.where(done)[0]:
                done_info = infos[env_idx]
                num_evacuated = int(done_info["num_evacuated"])
                last_episode_paths = done_info["paths"]
                last_episode_summary = {
                    "episode": episode_idx,
                    "env_index": int(env_idx),
                    "global_step": global_step,
                    "episode_reward": round(float(episode_reward[env_idx]), 6),
                    "episode_steps": int(episode_steps[env_idx]),
                    "num_evacuated": num_evacuated,
                    "arrival_steps": done_info["arrival_steps"].tolist(),
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
                is_better = num_evacuated > best_evacuated or (
                    num_evacuated == best_evacuated and episode_steps[env_idx] < best_steps
                )
                if is_better:
                    best_evacuated = num_evacuated
                    best_steps = int(episode_steps[env_idx])
                    best_episode_paths = [[p.copy() for p in path] for path in done_info["paths"]]
                    best_episode_summary = dict(last_episode_summary)
                    agent.save(str(save_dir / "best_model.pt"))

                episode_idx += 1
                episode_reward[env_idx] = 0.0
                episode_steps[env_idx] = 0
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

        if update % 5 == 0 or update == 1:
            recent = episode_logs[-10:]
            mean_reward = np.mean([row["episode_reward"] for row in recent]) if recent else 0.0
            mean_evac = np.mean([row["num_evacuated"] for row in recent]) if recent else 0.0
            print(
                f"update={update:04d}/{num_updates} step={global_step:07d} "
                f"loss={metrics['loss']:.4f} entropy={metrics['entropy']:.4f} "
                f"recent_reward={mean_reward:.2f} recent_evacuated={mean_evac:.2f}"
            )

    agent.save(str(save_dir / "final_model.pt"))
    write_csv(save_dir / "training_log.csv", episode_logs)
    if episode_logs:
        plot_training_curves(save_dir / "training_log.csv", save_dir / "training_curves.png")
    if last_episode_paths is None:
        last_episode_paths = infos[0].get("paths", [])
        last_episode_summary = {
            "episode": episode_idx,
            "global_step": global_step,
            "episode_reward": round(float(episode_reward[0]), 6),
            "episode_steps": int(episode_steps[0]),
            "num_evacuated": int(infos[0].get("num_evacuated", 0)),
            "arrival_steps": infos[0].get("arrival_steps", np.array([], dtype=np.int32)).tolist(),
        }

    if last_episode_paths:
        save_episode_outputs(
            save_dir,
            "last",
            env,
            last_episode_paths,
            last_episode_summary or {},
            not args.no_animation,
            args.animation_format,
            args.animation_fps,
        )
    if best_episode_paths:
        save_episode_outputs(
            save_dir,
            "best",
            env,
            best_episode_paths,
            best_episode_summary or {},
            not args.no_animation,
            args.animation_format,
            args.animation_fps,
        )
    print("Training finished.")


if __name__ == "__main__":
    main()
