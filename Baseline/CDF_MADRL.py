from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from mappo import MAPPOAgent, MAPPOConfig, RolloutBuffer, ValueNorm, apply_available_actions_mask, layer_init


TYPE_EMPTY = 0.0
TYPE_SELF = 1.0
TYPE_EXIT = 2.0
TYPE_EXIT_CONGESTION = 3.0
TYPE_NEIGHBOR = 4.0
TYPE_OBSTACLE = 5.0


@dataclass
class CDFObservationConfig:
    observed_agents: int = 35
    observation_range: int = 5
    max_obstacles: Optional[int] = None


@dataclass
class CDFRewardConfig:
    dynamic_threshold: int = 30
    arrival_reward: float = 30.0
    distance_reward_scale: float = 0.05
    global_reward_scale: float = 0.05
    max_steps: int = 300
    max_distance: float | None = None


class CDFObservationBuilder:
    """Builds the paper-style fixed-dimensional triplet observation."""

    def __init__(self, grid_map: np.ndarray, exits: np.ndarray, config: CDFObservationConfig):
        self.grid_map = np.asarray(grid_map, dtype=np.int8)
        self.exits = np.asarray(exits, dtype=np.float32).reshape(-1, 2)
        self.cfg = config
        self.rows, self.cols = self.grid_map.shape
        self.obstacle_offsets = self._build_obstacle_offsets()
        if self.cfg.max_obstacles is None:
            self.max_obstacles = len(self._diamond_offsets(include_center=False))
        else:
            self.max_obstacles = int(self.cfg.max_obstacles)
        self.num_triples = 1 + len(self.exits) + len(self.exits) + self.cfg.observed_agents + self.max_obstacles
        self.obs_dim = self.num_triples * 3

    def transform(self, infos) -> np.ndarray:
        return np.stack([self.transform_one(info) for info in infos], axis=0)

    def transform_one(self, info: dict) -> np.ndarray:
        positions = np.asarray(info["positions"], dtype=np.float32)
        active = np.asarray(info["active"], dtype=bool)
        congestions = self.exit_congestion(positions, active)
        obs = np.zeros((len(positions), self.obs_dim), dtype=np.float32)
        for agent_id, position in enumerate(positions):
            obs[agent_id] = self._agent_obs(agent_id, position, positions, active, congestions)
        return obs

    def exit_congestion(self, positions: np.ndarray, active: np.ndarray) -> np.ndarray:
        if len(self.exits) == 0:
            return np.zeros(0, dtype=np.float32)
        active_positions = positions[active]
        if len(active_positions) == 0:
            return np.zeros(len(self.exits), dtype=np.float32)
        deltas = np.abs(active_positions[:, None, :] - self.exits[None, :, :])
        manhattan = np.sum(deltas, axis=2)
        return np.sum(1.0 / (manhattan + 1.0), axis=0).astype(np.float32)

    def _agent_obs(
        self,
        agent_id: int,
        position: np.ndarray,
        positions: np.ndarray,
        active: np.ndarray,
        congestions: np.ndarray,
    ) -> np.ndarray:
        triples: list[Tuple[float, float, float]] = []
        row, col = float(position[0]), float(position[1])
        triples.append((row, col, TYPE_SELF))

        for exit_point in self.exits:
            triples.append((float(exit_point[0] - row), float(exit_point[1] - col), TYPE_EXIT))

        for congestion in congestions:
            triples.append((float(congestion), float(congestion), TYPE_EXIT_CONGESTION))

        triples.extend(self._neighbor_triples(agent_id, position, positions, active))
        triples.extend(self._obstacle_triples(position))

        if len(triples) < self.num_triples:
            triples.extend([(0.0, 0.0, TYPE_EMPTY)] * (self.num_triples - len(triples)))
        return np.asarray(triples[: self.num_triples], dtype=np.float32).reshape(-1)

    def _neighbor_triples(
        self,
        agent_id: int,
        position: np.ndarray,
        positions: np.ndarray,
        active: np.ndarray,
    ) -> list[Tuple[float, float, float]]:
        neighbors = []
        for other_id, other_pos in enumerate(positions):
            if other_id == agent_id or not active[other_id]:
                continue
            delta = other_pos - position
            distance = float(abs(delta[0]) + abs(delta[1]))
            if distance <= self.cfg.observation_range:
                neighbors.append((distance, float(delta[0]), float(delta[1])))
        neighbors.sort(key=lambda item: (item[0], item[1], item[2]))
        triples = [(dr, dc, TYPE_NEIGHBOR) for _, dr, dc in neighbors[: self.cfg.observed_agents]]
        triples.extend([(0.0, 0.0, TYPE_EMPTY)] * (self.cfg.observed_agents - len(triples)))
        return triples

    def _obstacle_triples(self, position: np.ndarray) -> list[Tuple[float, float, float]]:
        row, col = int(position[0]), int(position[1])
        triples = []
        for dr, dc in self.obstacle_offsets:
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols or self.grid_map[rr, cc] == 1:
                triples.append((float(dr), float(dc), TYPE_OBSTACLE))
                if len(triples) >= self.max_obstacles:
                    break
        triples.extend([(0.0, 0.0, TYPE_EMPTY)] * (self.max_obstacles - len(triples)))
        return triples

    def _build_obstacle_offsets(self) -> list[Tuple[int, int]]:
        offsets = self._diamond_offsets(include_center=False)
        offsets.sort(key=lambda item: (abs(item[0]) + abs(item[1]), item[0], item[1]))
        return offsets

    def _diamond_offsets(self, include_center: bool) -> list[Tuple[int, int]]:
        beta = int(self.cfg.observation_range)
        offsets = []
        for dr in range(-beta, beta + 1):
            for dc in range(-beta, beta + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if abs(dr) + abs(dc) <= beta:
                    offsets.append((dr, dc))
        return offsets


class CDFActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.policy_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.policy_head(self.net(obs))
        return apply_available_actions_mask(logits, available_actions)

    def distribution(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> Categorical:
        return Categorical(logits=self.forward(obs, available_actions))


class CDFCentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim: int, num_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.global_obs_dim = int(global_obs_dim)
        self.num_agents = int(num_agents)
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(self.global_obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, self.num_agents), std=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
        if global_obs.shape[-1] != self.global_obs_dim:
            raise ValueError(f"Expected global_obs dim {self.global_obs_dim}, got {tuple(global_obs.shape)}.")
        return self.value_head(global_obs)


class CDFMADRLAgent(MAPPOAgent):
    """CDF-MADRL with the paper-style FCN actor and centralized FCN critic."""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: MAPPOConfig):
        self.cfg = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_obs_dim = obs_dim * num_agents
        self.device = config.device

        self.actor = CDFActor(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = CDFCentralizedCritic(self.global_obs_dim, num_agents, config.hidden_dim).to(self.device)
        self.value_normalizer = (
            ValueNorm(1, beta=config.value_norm_beta, epsilon=config.value_norm_epsilon, device=self.device)
            if config.use_valuenorm
            else None
        )
        actor_lr = config.actor_lr if config.actor_lr is not None else config.learning_rate
        critic_lr = config.critic_lr if config.critic_lr is not None else config.learning_rate
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1.0e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1.0e-5)


class CDFRewardModulator:
    """Implements the paper's Eq. (11)-(13) reward terms on the MAPPO reward scale."""

    def __init__(self, config: CDFRewardConfig | None = None):
        self.cfg = config or CDFRewardConfig()
        self.steps: np.ndarray | None = None

    def reset(self, infos, exits) -> None:
        self.steps = np.zeros(len(infos), dtype=np.int32)

    def reset_at(self, env_index: int) -> None:
        if self.steps is not None:
            self.steps[int(env_index)] = 0

    def to_dict(self) -> dict:
        return asdict(self.cfg)

    def shape(self, rewards, prev_infos, next_infos, exits) -> np.ndarray:
        base = np.asarray(rewards, dtype=np.float32)
        paper_rewards = np.zeros_like(base, dtype=np.float32)
        exits_arr = np.asarray(exits, dtype=np.float32).reshape(-1, 2)
        if self.steps is None or len(self.steps) != len(next_infos):
            self.reset(next_infos, exits_arr)

        for env_idx, (prev_info, next_info) in enumerate(zip(prev_infos, next_infos)):
            if self.steps is not None:
                self.steps[env_idx] += 1
                step = int(self.steps[env_idx])
            else:
                step = 0
            paper_rewards[env_idx] = self._one_env_reward(prev_info, next_info, exits_arr, step)
        return paper_rewards

    def _one_env_reward(
        self,
        prev_info: dict,
        next_info: dict,
        exits: np.ndarray,
        step: int,
    ) -> np.ndarray:
        prev_pos = np.asarray(prev_info["positions"], dtype=np.float32)
        prev_active = np.asarray(prev_info["active"], dtype=bool)
        next_active = np.asarray(next_info["active"], dtype=bool)
        active_count = int(np.sum(prev_active))
        congestions = self._exit_congestion(prev_pos, prev_active, exits)
        rewards = np.zeros(len(prev_pos), dtype=np.float32)

        for agent_id in range(len(prev_pos)):
            if not prev_active[agent_id]:
                rewards[agent_id] = 0.0
                continue

            arrived_now = not next_active[agent_id]
            if arrived_now:
                rewards[agent_id] = float(self.cfg.arrival_reward)
                continue

            distances = np.sum(np.abs(exits - prev_pos[agent_id]), axis=1)
            distance_scale = self._distance_scale(prev_pos, exits)
            distance_terms = distances / distance_scale
            congestion_terms = congestions / max(1.0, float(active_count))
            if active_count < self.cfg.dynamic_threshold:
                objective = distance_terms
            else:
                objective = distance_terms + congestion_terms
            distance_reward = self._penalty_unit * float(np.min(objective))
            global_reward = self._global_reward(step, active_count, len(prev_pos))
            rewards[agent_id] = distance_reward + global_reward
        return rewards

    @property
    def _penalty_unit(self) -> float:
        return -abs(float(self.cfg.distance_reward_scale))

    def _global_reward(self, step: int, active_count: int, num_agents: int) -> float:
        time_term = min(1.0, max(0.0, float(step) / max(1.0, float(self.cfg.max_steps))))
        active_term = float(active_count) / max(1.0, float(num_agents))
        return -abs(float(self.cfg.global_reward_scale)) * (time_term + active_term)

    def _distance_scale(self, positions: np.ndarray, exits: np.ndarray) -> float:
        configured = self.cfg.max_distance
        if configured is not None and float(configured) > 0.0:
            return max(1.0, float(configured))
        points = np.vstack([positions[:, :2], exits[:, :2]])
        extent = np.ptp(points, axis=0)
        return max(1.0, float(np.sum(extent)))

    def _exit_congestion(self, positions: np.ndarray, active: np.ndarray, exits: np.ndarray) -> np.ndarray:
        active_positions = positions[active]
        if len(active_positions) == 0:
            return np.zeros(len(exits), dtype=np.float32)
        manhattan = np.sum(np.abs(active_positions[:, None, :] - exits[None, :, :]), axis=2)
        return np.sum(1.0 / (manhattan + 1.0), axis=0).astype(np.float32)
