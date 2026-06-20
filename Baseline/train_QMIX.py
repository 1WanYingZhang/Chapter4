from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from QMIX import EpisodeReplayBuffer, QMIXAgent, QMIXConfig
from train import (
    _points_to_arg,
    agent_arrival_record_summary,
    build_algorithm_save_dir,
    build_env,
    compute_episode_metrics,
    init_agent_arrival_records,
    reset_agent_arrival_record,
    save_agent_metrics_csv,
    save_episode_outputs,
    save_initial_map,
    save_run_info_txt,
    set_random_seed,
    update_agent_arrival_records,
)
from utils import write_csv
from visualize import plot_training_curves


# =============================================================================
# QMIX training entry.
# Edit this file, then run:
#
#     python train_QMIX.py
#
# The QMIX algorithm implementation is in QMIX.py.
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
LOCAL_VIEW_RADIUS = 5
MAX_NEIGHBORS = 4
EXIT_CAPACITY = 1
NO_WAIT = False
SAVE_ANIMATION = False
ANIMATION_FORMAT = "mp4"
ANIMATION_FPS = 8
SAVE_DIR = None

BUFFER_SIZE = 5000
BATCH_SIZE = 32
TRAIN_START_EPISODES = 32
TRAIN_INTERVAL_EPISODES = 1
# PyMARL trains once after each collected episode when the replay buffer can sample a batch.
GRADIENT_STEPS_PER_EPISODE = 1
TARGET_UPDATE_INTERVAL_EPISODES = 200
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50_000


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
        seed=SEED,
        save_dir=SAVE_DIR,
        no_animation=not SAVE_ANIMATION,
        animation_format=ANIMATION_FORMAT,
        animation_fps=ANIMATION_FPS,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        train_start_episodes=TRAIN_START_EPISODES,
        train_interval_episodes=TRAIN_INTERVAL_EPISODES,
        gradient_steps_per_episode=GRADIENT_STEPS_PER_EPISODE,
        target_update_interval_episodes=TARGET_UPDATE_INTERVAL_EPISODES,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
    )


def _one_hot(actions: np.ndarray, action_dim: int) -> np.ndarray:
    out = np.zeros((len(actions), action_dim), dtype=np.float32)
    out[np.arange(len(actions)), actions.astype(np.int64)] = 1.0
    return out


def _copy_env_paths(env) -> list[list[np.ndarray]]:
    return [[point.copy() for point in path] for path in env.paths]


def collect_episode(
    env,
    agent: QMIXAgent,
    epsilon: float,
    seed: int | None = None,
    training_started_at: dt.datetime | None = None,
) -> tuple[dict[str, np.ndarray], dict, list[list[np.ndarray]], dict]:
    obs, info = env.reset(seed=seed)
    prev_actions = np.zeros((env.n_agents, env.n_actions), dtype=np.float32)
    hidden = agent.agent.init_hidden(env.n_agents, agent.device)

    obs_seq = [obs.copy()]
    state_seq = [obs.reshape(-1).copy()]
    avail_seq = [info.get("available_actions", env.available_actions()).copy()]
    actions_seq = []
    rewards_seq = []
    terminated_seq = []
    collision_counts = np.zeros(env.n_agents, dtype=np.int32)
    arrival_time_seconds, exit_positions = init_agent_arrival_records(1, env.n_agents)
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < env.cfg.max_steps:
        actions, hidden = agent.select_actions(obs, avail_seq[-1], prev_actions, hidden, epsilon)
        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = bool(terminated or truncated)
        for agent_id, component in enumerate(next_info.get("reward_components", [])):
            if component.get("collision", 0.0) < 0.0:
                collision_counts[agent_id] += 1
        update_agent_arrival_records(arrival_time_seconds, exit_positions, [next_info], training_started_at)

        actions_seq.append(actions.copy())
        rewards_seq.append(float(np.sum(rewards)))
        terminated_seq.append(float(terminated))
        total_reward += float(np.sum(rewards))
        obs = next_obs
        info = next_info
        prev_actions = _one_hot(actions, env.n_actions)
        obs_seq.append(obs.copy())
        state_seq.append(obs.reshape(-1).copy())
        avail_seq.append(info.get("available_actions", env.available_actions()).copy())
        steps += 1

    episode = {
        "obs": np.asarray(obs_seq, dtype=np.float32),
        "states": np.asarray(state_seq, dtype=np.float32),
        "avail_actions": np.asarray(avail_seq, dtype=np.float32),
        "actions": np.asarray(actions_seq, dtype=np.int64),
        "rewards": np.asarray(rewards_seq, dtype=np.float32),
        "terminated": np.asarray(terminated_seq, dtype=np.float32),
    }
    summary = {
        "episode_reward": round(float(total_reward), 6),
        "episode_steps": int(steps),
        "num_evacuated": int(info.get("num_evacuated", 0)),
        "arrival_steps": info.get("arrival_steps", np.array([], dtype=np.int32)).tolist(),
        "collision_counts": collision_counts.astype(int).tolist(),
        "average_collision_count": round(float(np.mean(collision_counts)), 6),
        **agent_arrival_record_summary(arrival_time_seconds, exit_positions, 0),
    }
    return episode, info, info.get("paths", []), summary


def main(seed_override: int | None = None) -> None:
    args = make_code_config_args()
    if seed_override is not None:
        args.seed = int(seed_override)
    training_started_at = dt.datetime.now()
    set_random_seed(args.seed)
    env = build_env(args)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = build_algorithm_save_dir(args, "QMIX", run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = QMIXConfig(
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_start_episodes=args.train_start_episodes,
        train_interval_episodes=args.train_interval_episodes,
        gradient_steps_per_episode=args.gradient_steps_per_episode,
        target_update_interval_episodes=args.target_update_interval_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    agent = QMIXAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)
    replay = EpisodeReplayBuffer(cfg.buffer_size)

    print("Algorithm: QMIX")
    print(f"Device: {cfg.device}")
    print(f"Seed: {args.seed}")
    print(f"Map: {env.rows}x{env.cols}, agents={env.n_agents}, exits={len(env.exits)}, actions={env.n_actions}")
    print(f"Exits: {[tuple(x) for x in env.exits.tolist()]}")
    print(f"Starts: {[tuple(x) for x in env.starts.tolist()]}")
    print(f"Save dir: {save_dir}")
    save_run_info_txt(save_dir, args, env, training_started_at=training_started_at)
    initial_map_path = save_initial_map(save_dir, env, args.map_file)
    print(f"Initial map: {initial_map_path}")

    global_step = 0
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
    last_train_metrics = {}

    while global_step < cfg.total_timesteps:
        episode_started_at = dt.datetime.now()
        epsilon = agent.epsilon(global_step)
        episode, info, paths, summary = collect_episode(
            env,
            agent,
            epsilon,
            seed=args.seed + episode_idx,
            training_started_at=training_started_at,
        )
        replay.add(episode)
        global_step += int(summary["episode_steps"])

        summary.update(
            {
                "episode": episode_idx,
                "global_step": global_step,
                "epsilon": round(float(epsilon), 6),
            }
        )
        num_evacuated = int(summary["num_evacuated"])
        episode_logs.append(
            {
                "episode": episode_idx,
                "env_index": 0,
                "global_step": global_step,
                "episode_reward": summary["episode_reward"],
                "episode_steps": summary["episode_steps"],
                "num_evacuated": num_evacuated,
            }
        )

        fully_evacuated = num_evacuated == env.n_agents and bool(paths)
        is_model_better = num_evacuated > best_evacuated or (
            num_evacuated == best_evacuated and summary["episode_steps"] < best_steps
        )
        if is_model_better:
            best_evacuated = num_evacuated
            best_steps = int(summary["episode_steps"])
            agent.save(save_dir / "best_model.pt")
        if fully_evacuated:
            last_episode_paths = paths
            last_episode_summary = dict(summary)
            is_better = best_episode_paths is None or summary["episode_steps"] < best_path_steps
            if is_better:
                best_path_steps = int(summary["episode_steps"])
                best_episode_paths = [[p.copy() for p in path] for path in paths]
                best_episode_summary = dict(summary)
        elif best_episode_paths is None:
            fallback_episode_paths = _copy_env_paths(env)
            fallback_episode_summary = dict(summary)

        if (
            len(replay) >= cfg.train_start_episodes
            and episode_idx % cfg.train_interval_episodes == 0
            and len(replay) >= cfg.batch_size
        ):
            train_metrics = [agent.train_step(replay) for _ in range(max(1, cfg.gradient_steps_per_episode))]
            last_train_metrics = {
                key: float(np.mean([item.get(key, 0.0) for item in train_metrics]))
                for key in train_metrics[-1]
            }

        if episode_idx > 0 and episode_idx % cfg.target_update_interval_episodes == 0:
            agent.update_targets()

        episode_seconds = (dt.datetime.now() - episode_started_at).total_seconds()
        recent = episode_logs[-10:]
        mean_reward = np.mean([row["episode_reward"] for row in recent]) if recent else 0.0
        mean_evac = np.mean([row["num_evacuated"] for row in recent]) if recent else 0.0
        mean_steps = np.mean([row["episode_steps"] for row in recent]) if recent else 0.0
        training_logs.append(
            {
                "episode": episode_idx,
                "global_step": global_step,
                "episode_time_seconds": round(float(episode_seconds), 6),
                "episodes_completed": episode_idx + 1,
                "epsilon": round(float(epsilon), 6),
                "recent_reward": round(float(mean_reward), 6),
                "recent_evacuated": round(float(mean_evac), 6),
                "recent_episode_steps": round(float(mean_steps), 6),
                "loss": round(float(last_train_metrics.get("loss", 0.0)), 6),
                "q_total": round(float(last_train_metrics.get("q_total", 0.0)), 6),
            }
        )
        print(
            f"episode={episode_idx:05d} step={global_step:07d}/{cfg.total_timesteps} "
            f"episode_time={episode_seconds:.2f}s "
            f"epsilon={epsilon:.3f} loss={last_train_metrics.get('loss', 0.0):.4f} "
            f"q_total={last_train_metrics.get('q_total', 0.0):.3f} "
            f"recent_reward={mean_reward:.2f} recent_evacuated={mean_evac:.2f}"
        )

        episode_idx += 1

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

    if last_episode_paths is None and fallback_episode_paths is not None:
        last_episode_paths = fallback_episode_paths
        last_episode_summary = fallback_episode_summary

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
    print("QMIX training finished.")


def run_all_seeds() -> None:
    for seed in RUN_SEEDS:
        print(f"\n===== Running seed {seed} =====")
        main(seed_override=seed)


if __name__ == "__main__":
    run_all_seeds()
