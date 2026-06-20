from __future__ import annotations

import datetime as dt
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from IPPO import IPPOAgent, IPPOConfig, IPPORolloutBuffer
from train import (
    _points_to_arg,
    agent_arrival_record_summary,
    build_algorithm_save_dir,
    build_env,
    compute_episode_metrics,
    freeze_shared_starts,
    init_agent_arrival_records,
    make_update_training_log_row,
    reset_agent_arrival_record,
    save_agent_metrics_csv,
    save_episode_outputs,
    save_initial_map,
    save_run_info_txt,
    set_random_seed,
    update_agent_arrival_records,
)
from utils import write_csv
from vec_env import make_vec_env
from visualize import plot_training_curves


# =============================================================================
# IPPO training entry.
# Edit this file, then run:
#
#     python train_IPPO.py
#
# The IPPO algorithm implementation is in IPPO.py.
# =============================================================================
USE_CODE_CONFIG = True

MAP_FILE = Path(__file__).resolve().parent / "狭窄地图.txt"

# Use [] to read exits marked as 2 in the map txt automatically.
EXITS: list[tuple[int, int]] = []

NUM_AGENTS = 30

# Use [] to randomly sample starts according to NUM_AGENTS and SEED.
STARTS: list[tuple[int, int]] = []

SEED = 1
RUN_SEEDS = [1, 2, 3, 4, 5]

MAX_STEPS = 500
TOTAL_TIMESTEPS = 204800
ROLLOUT_STEPS = 128
UPDATE_EPOCHS = 8
MINIBATCH_SIZE = 512
NUM_ENVS = 8
VEC_ENV_TYPE = "subproc"
SHARE_PARAMETERS = True
LOCAL_VIEW_RADIUS = 5
MAX_NEIGHBORS = 4
EXIT_CAPACITY = 1
NO_WAIT = False
SAVE_ANIMATION = False
ANIMATION_FORMAT = "mp4"
ANIMATION_FPS = 8
SAVE_DIR = None


def make_code_config_args() -> SimpleNamespace:
    return SimpleNamespace(
        map_file=str(MAP_FILE),
        num_agents=NUM_AGENTS,
        starts=_points_to_arg(STARTS),
        exits=_points_to_arg(EXITS),
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
    args = make_code_config_args()
    if seed_override is not None:
        args.seed = int(seed_override)
    training_started_at = dt.datetime.now()
    set_random_seed(args.seed)
    if args.num_envs < 1:
        raise ValueError("NUM_ENVS must be at least 1.")

    args = freeze_shared_starts(args, build_env)
    vec_env = make_vec_env([partial(build_env, args) for _ in range(args.num_envs)], args.vec_env_type)
    env = vec_env.envs[0]
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = build_algorithm_save_dir(args, "IPPO", run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = IPPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        share_parameters=SHARE_PARAMETERS,
    )
    agent = IPPOAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)

    if cfg.share_parameters:
        print("Algorithm: IPPO (decentralized actor-critic with shared parameters)")
    else:
        print("Algorithm: IPPO (separate decentralized actor-critic per agent)")
    print(f"Device: {cfg.device}")
    print(f"Parameter sharing: {cfg.share_parameters}")
    print(f"Seed: {args.seed}")
    print(f"Map: {env.rows}x{env.cols}, agents={env.n_agents}, exits={len(env.exits)}, actions={env.n_actions}")
    print(f"Exits: {[tuple(x) for x in env.exits.tolist()]}")
    print(f"Starts: {[tuple(x) for x in env.starts.tolist()]}")
    print(f"Save dir: {save_dir}")
    save_run_info_txt(save_dir, args, env, training_started_at=training_started_at)
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

    for update in range(1, num_updates + 1):
        update_started_at = dt.datetime.now()
        buffer = IPPORolloutBuffer(
            cfg.rollout_steps,
            vec_env.num_envs,
            env.n_agents,
            env.obs_dim,
            env.n_actions,
            cfg.device,
        )
        for _ in range(cfg.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
            active_mask_tensor = torch.tensor(
                np.stack([item["active"] for item in infos], axis=0),
                dtype=torch.float32,
                device=cfg.device,
            )
            available_actions_tensor = torch.tensor(
                np.stack([item["available_actions"] for item in infos], axis=0),
                dtype=torch.float32,
                device=cfg.device,
            )

            actions, logprobs, values = agent.act(obs_tensor, available_actions_tensor)
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

            buffer.add(
                obs_tensor,
                actions,
                logprobs,
                reward_tensor,
                done_tensor,
                values,
                active_mask_tensor,
                available_actions_tensor,
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
                    agent.save(save_dir / "best_model.pt")
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
        with torch.no_grad():
            last_values = agent.values(obs_tensor)
        buffer.compute_returns_and_advantages(last_values, cfg.gamma, cfg.gae_lambda)
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
            f"loss={metrics.get('loss', 0.0):.4f} entropy={metrics.get('entropy', 0.0):.4f} "
            f"recent_reward={mean_reward:.2f} recent_evacuated={mean_evac:.2f}"
        )

    agent.save(save_dir / "final_model.pt")
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
    print("IPPO training finished.")


def run_all_seeds() -> None:
    for seed in RUN_SEEDS:
        print(f"\n===== Running seed {seed} =====")
        main(seed_override=seed)


if __name__ == "__main__":
    run_all_seeds()
