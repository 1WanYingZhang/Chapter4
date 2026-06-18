from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from emappo import MAPPOAgent, MAPPOConfig
from emappo_env import EvacuationConfig, MultiAgentGridEvacuationEnv
from train import build_env
from visualize import animate_paths, plot_paths


# =============================================================================
# Code config: edit here, then run python evaluate.py directly.
# If MODEL_PATH is None, the newest best_model.pt/final_model.pt under results is used.
# Empty EXITS reads target cells marked as 2 in the map txt.
# Empty STARTS randomly samples start cells according to NUM_AGENTS and SEED.
# =============================================================================
USE_CODE_CONFIG = True
MODEL_PATH: str | Path | None = None
MAP_FILE = Path(__file__).resolve().parent / "普通地图.txt"
NUM_AGENTS = 3
STARTS: list[tuple[int, int]] = []
EXITS: list[tuple[int, int]] = []
MAX_STEPS = 300
LOCAL_VIEW_RADIUS = 5
MAX_NEIGHBORS = 4
EXIT_CAPACITY = 1
NO_WAIT = False
SAVE_PATH: str | Path | None = None
SAVE_ANIMATION = False
ANIMATION_FORMAT = "mp4"
ANIMATION_FPS = 8
SEED = 1


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _points_to_arg(points: list[tuple[int, int]] | str | None) -> str | None:
    if points is None or points == "":
        return None
    if isinstance(points, str):
        return points or None
    if len(points) == 0:
        return None
    return ";".join(f"{row},{col}" for row, col in points)


def _latest_model_path() -> Path:
    results_dir = Path(__file__).resolve().parent / "results"
    candidates = list(results_dir.glob("**/best_model.pt")) + list(results_dir.glob("**/final_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No model found. Set MODEL_PATH in evaluate.py or train a model first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPPO evacuation policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--map-file", type=str, default=str(MAP_FILE))
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--starts", type=str, default=None)
    parser.add_argument("--exits", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--local-view-radius", type=int, default=LOCAL_VIEW_RADIUS)
    parser.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    parser.add_argument("--exit-capacity", type=int, default=EXIT_CAPACITY)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-format", choices=["mp4", "gif"], default=ANIMATION_FORMAT)
    parser.add_argument("--animation-fps", type=int, default=ANIMATION_FPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def make_code_config_args() -> SimpleNamespace:
    model_path = Path(MODEL_PATH) if MODEL_PATH is not None else _latest_model_path()
    return SimpleNamespace(
        model=str(model_path),
        map_file=str(MAP_FILE),
        num_agents=NUM_AGENTS,
        starts=_points_to_arg(STARTS),
        exits=_points_to_arg(EXITS),
        max_steps=MAX_STEPS,
        local_view_radius=LOCAL_VIEW_RADIUS,
        max_neighbors=MAX_NEIGHBORS,
        exit_capacity=EXIT_CAPACITY,
        no_wait=NO_WAIT,
        save_path=str(SAVE_PATH) if SAVE_PATH is not None else None,
        save_animation=SAVE_ANIMATION,
        animation_format=ANIMATION_FORMAT,
        animation_fps=ANIMATION_FPS,
        seed=SEED,
    )


def main() -> None:
    args = make_code_config_args() if USE_CODE_CONFIG else make_parser().parse_args()

    set_random_seed(args.seed)
    env: MultiAgentGridEvacuationEnv = build_env(args)
    print(f"Use code config: {USE_CODE_CONFIG}")
    print(f"Seed: {args.seed}")
    print(f"Model: {Path(args.model).resolve()}")
    print(f"Exits: {[tuple(x) for x in env.exits.tolist()]}")
    print(f"Starts: {[tuple(x) for x in env.starts.tolist()]}")
    cfg = MAPPOConfig()
    agent = MAPPOAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)
    agent.load(args.model)

    obs, info = env.reset()
    total_reward = np.zeros(env.n_agents, dtype=np.float32)
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
        available_actions_tensor = torch.tensor(
            info.get("available_actions", env.available_actions()),
            dtype=torch.float32,
            device=cfg.device,
        )
        with torch.no_grad():
            actions = agent.greedy_action(obs_tensor, available_actions_tensor)
        obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        total_reward += rewards
        done = terminated or truncated

    save_path = Path(args.save_path or Path(args.model).with_name("evaluation_paths.png"))
    plot_paths(env.grid_map, info["paths"], env.exits, save_path, map_file=args.map_file)
    if args.save_animation:
        animation_path = animate_paths(
            env.grid_map,
            info["paths"],
            env.exits,
            save_path.with_name(save_path.stem + f"_animation.{args.animation_format}"),
            fps=args.animation_fps,
            map_file=args.map_file,
        )
        print(f"Saved animation: {animation_path}")
    print(f"Evacuated: {info['num_evacuated']}/{env.n_agents}")
    print(f"Arrival steps: {info['arrival_steps'].tolist()}")
    print(f"Total reward: {total_reward.round(3).tolist()}")
    print(f"Saved path plot: {save_path}")


if __name__ == "__main__":
    main()
