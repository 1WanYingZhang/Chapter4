from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass
class MAPPOConfig:
    total_timesteps: int = 50_000
    rollout_steps: int = 1024
    update_epochs: int = 8
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim: int, num_agents: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_agents), std=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)


class RolloutBuffer:
    def __init__(self, steps: int, num_agents: int, obs_dim: int, global_obs_dim: int, device: str):
        self.steps = steps
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((steps, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.global_obs = torch.zeros((steps, global_obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((steps, num_agents), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_agents), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if self.ptr >= self.steps:
            raise RuntimeError("RolloutBuffer is full.")
        self.obs[self.ptr].copy_(obs)
        self.global_obs[self.ptr].copy_(global_obs)
        self.actions[self.ptr].copy_(actions)
        self.logprobs[self.ptr].copy_(logprobs)
        self.rewards[self.ptr].copy_(rewards)
        self.dones[self.ptr].copy_(dones)
        self.values[self.ptr].copy_(values)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values


class MAPPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: MAPPOConfig):
        self.cfg = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_obs_dim = obs_dim * num_agents
        self.device = config.device

        self.actor = DiscreteActor(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = CentralizedCritic(self.global_obs_dim, num_agents, config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor, global_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.distribution(obs)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        values = self.critic(global_obs.unsqueeze(0)).squeeze(0)
        return actions, logprobs, values

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        return torch.argmax(logits, dim=-1)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        steps = buffer.steps
        n = self.num_agents
        total_samples = steps * n

        b_obs = buffer.obs.reshape(total_samples, self.obs_dim)
        b_global = (
            buffer.global_obs[:, None, :]
            .expand(steps, n, self.global_obs_dim)
            .reshape(total_samples, self.global_obs_dim)
        )
        b_actions = buffer.actions.reshape(total_samples)
        b_logprobs = buffer.logprobs.reshape(total_samples)
        b_advantages = buffer.advantages.reshape(total_samples)
        b_returns = buffer.returns.reshape(total_samples)
        b_agent_ids = torch.arange(n, device=self.device).repeat(steps)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        batch_size = total_samples
        minibatch_size = min(self.cfg.minibatch_size, batch_size)

        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        update_count = 0
        for _ in range(self.cfg.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                dist = self.actor.distribution(b_obs[mb_idx])
                new_logprob = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()

                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                values_all = self.critic(b_global[mb_idx])
                values = values_all.gather(1, b_agent_ids[mb_idx].unsqueeze(1)).squeeze(1)
                value_loss = 0.5 * ((values - b_returns[mb_idx]) ** 2).mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["loss"] += float(loss.item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                update_count += 1

        for key in metrics:
            metrics[key] /= max(1, update_count)
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "num_agents": self.num_agents,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
