from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mappo import MAPPOAgent, MAPPOConfig
from mappo_env import EvacuationConfig, MultiAgentGridEvacuationEnv
from train import build_env
from visualize import animate_paths, plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPPO evacuation policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--map-file", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--starts", type=str, default=None)
    parser.add_argument("--exits", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--local-view-radius", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--exit-capacity", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--animation-fps", type=int, default=8)
    args = parser.parse_args()

    env: MultiAgentGridEvacuationEnv = build_env(args)
    cfg = MAPPOConfig()
    agent = MAPPOAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)
    agent.load(args.model)

    obs, info = env.reset()
    total_reward = np.zeros(env.n_agents, dtype=np.float32)
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
        with torch.no_grad():
            actions = agent.greedy_action(obs_tensor)
        obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        total_reward += rewards
        done = terminated or truncated

    save_path = Path(args.save_path or Path(args.model).with_name("evaluation_paths.png"))
    plot_paths(env.grid_map, info["paths"], env.exits, save_path)
    if args.save_animation:
        animation_path = animate_paths(
            env.grid_map,
            info["paths"],
            env.exits,
            save_path.with_name(save_path.stem + f"_animation.{args.animation_format}"),
            fps=args.animation_fps,
        )
        print(f"Saved animation: {animation_path}")
    print(f"Evacuated: {info['num_evacuated']}/{env.n_agents}")
    print(f"Arrival steps: {info['arrival_steps'].tolist()}")
    print(f"Total reward: {total_reward.round(3).tolist()}")
    print(f"Saved path plot: {save_path}")


if __name__ == "__main__":
    main()
