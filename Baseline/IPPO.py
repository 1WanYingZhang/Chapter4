from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from mappo import ActorFeatureEncoder, CriticFeatureEncoder, apply_available_actions_mask, huber_loss, layer_init, masked_mean


# IPPO is fully decentralized here: each agent owns its own actor, critic,
# optimizer, and local-observation PPO update.
@dataclass
class IPPOConfig:
    total_timesteps: int = 200_000
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
    hidden_dim: int = 256
    share_parameters: bool = True
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LocalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.actor_encoder = ActorFeatureEncoder(
            obs_dim=obs_dim,
            map_repr_dim=256,
            goal_repr_dim=32,
            hidden_dim=hidden_dim,
        )
        self.actor_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic_encoder = CriticFeatureEncoder(
            obs_dim=obs_dim,
            map_repr_dim=128,
            goal_repr_dim=16,
            agent_feature_dim=128,
        )
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(128, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def distribution(self, obs: torch.Tensor, available_actions: torch.Tensor | None = None) -> Categorical:
        logits = apply_available_actions_mask(self.actor_head(self.actor_encoder(obs)), available_actions)
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_encoder(obs)).squeeze(-1)


class IPPORolloutBuffer:
    def __init__(self, steps: int, num_envs: int, num_agents: int, obs_dim: int, action_dim: int, device: str):
        self.steps = int(steps)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.obs = torch.zeros((steps, num_envs, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((steps, num_envs, num_agents), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.active_masks = torch.ones((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.available_actions = torch.zeros((steps, num_envs, num_agents, action_dim), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        active_masks: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> None:
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr].copy_(actions)
        self.logprobs[self.ptr].copy_(logprobs)
        self.rewards[self.ptr].copy_(rewards)
        self.dones[self.ptr].copy_(dones)
        self.values[self.ptr].copy_(values)
        self.active_masks[self.ptr].copy_(active_masks)
        self.available_actions[self.ptr].copy_(available_actions)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        last_gae = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.steps)):
            next_values = last_values if t == self.steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values


class IPPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, cfg: IPPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.share_parameters = bool(cfg.share_parameters)
        if self.share_parameters:
            self.shared_policy = LocalActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
            self.policies = nn.ModuleList([self.shared_policy])
            self.optimizers = [optim.Adam(self.shared_policy.parameters(), lr=cfg.learning_rate, eps=1e-5)]
        else:
            self.shared_policy = None
            self.policies = nn.ModuleList(
                [LocalActorCritic(obs_dim, action_dim, cfg.hidden_dim) for _ in range(num_agents)]
            ).to(self.device)
            self.optimizers = [
                optim.Adam(self.policies[agent_id].parameters(), lr=cfg.learning_rate, eps=1e-5)
                for agent_id in range(num_agents)
            ]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, available_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.share_parameters:
            flat_obs = obs.reshape(-1, self.obs_dim)
            flat_available = available_actions.reshape(-1, self.action_dim)
            dist = self.shared_policy.distribution(flat_obs, flat_available)
            flat_actions = dist.sample()
            flat_logprobs = dist.log_prob(flat_actions)
            flat_values = self.shared_policy.value(flat_obs)
            return (
                flat_actions.reshape(obs.shape[0], self.num_agents),
                flat_logprobs.reshape(obs.shape[0], self.num_agents),
                flat_values.reshape(obs.shape[0], self.num_agents),
            )

        actions = torch.zeros((obs.shape[0], self.num_agents), dtype=torch.long, device=self.device)
        logprobs = torch.zeros((obs.shape[0], self.num_agents), dtype=torch.float32, device=self.device)
        values = torch.zeros((obs.shape[0], self.num_agents), dtype=torch.float32, device=self.device)
        for agent_id, policy in enumerate(self.policies):
            dist = policy.distribution(obs[:, agent_id], available_actions[:, agent_id])
            action = dist.sample()
            actions[:, agent_id] = action
            logprobs[:, agent_id] = dist.log_prob(action)
            values[:, agent_id] = policy.value(obs[:, agent_id])
        return actions, logprobs, values

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor, available_actions: torch.Tensor) -> torch.Tensor:
        if self.share_parameters:
            flat_obs = obs.reshape(-1, self.obs_dim)
            flat_available = available_actions.reshape(-1, self.action_dim)
            logits = apply_available_actions_mask(
                self.shared_policy.actor_head(self.shared_policy.actor_encoder(flat_obs)),
                flat_available,
            )
            return torch.argmax(logits, dim=-1).reshape(obs.shape[0], self.num_agents)

        actions = torch.zeros((obs.shape[0], self.num_agents), dtype=torch.long, device=self.device)
        for agent_id, policy in enumerate(self.policies):
            logits = apply_available_actions_mask(
                policy.actor_head(policy.actor_encoder(obs[:, agent_id])),
                available_actions[:, agent_id],
            )
            actions[:, agent_id] = torch.argmax(logits, dim=-1)
        return actions

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_parameters:
            flat_obs = obs.reshape(-1, self.obs_dim)
            return self.shared_policy.value(flat_obs).reshape(obs.shape[0], self.num_agents)
        return torch.stack(
            [policy.value(obs[:, agent_id]) for agent_id, policy in enumerate(self.policies)],
            dim=1,
        )

    def update(self, buffer: IPPORolloutBuffer) -> dict:
        cfg = self.cfg
        if self.share_parameters:
            policy = self.shared_policy
            optimizer = self.optimizers[0]
            b_obs = buffer.obs.reshape(-1, buffer.obs_dim)
            b_actions = buffer.actions.reshape(-1)
            b_logprobs = buffer.logprobs.reshape(-1).detach()
            b_returns = buffer.returns.reshape(-1).detach()
            b_advantages = buffer.advantages.reshape(-1).detach()
            b_active = buffer.active_masks.reshape(-1)
            b_available = buffer.available_actions.reshape(-1, buffer.action_dim)

            active_count = b_active.sum().clamp_min(1.0)
            adv_mean = (b_advantages * b_active).sum() / active_count
            adv_var = ((b_advantages - adv_mean).pow(2) * b_active).sum() / active_count
            b_advantages = (b_advantages - adv_mean) / torch.sqrt(adv_var + 1e-8)

            batch_size = b_obs.shape[0]
            minibatch_size = min(cfg.minibatch_size, batch_size)
            inds = np.arange(batch_size)
            metrics = {}
            for _ in range(cfg.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = torch.as_tensor(inds[start : start + minibatch_size], dtype=torch.long, device=self.device)
                    dist = policy.distribution(b_obs[mb_inds], b_available[mb_inds])
                    new_logprobs = dist.log_prob(b_actions[mb_inds])
                    entropy = dist.entropy()
                    new_values = policy.value(b_obs[mb_inds])

                    logratio = new_logprobs - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb_inds]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    pg_loss = masked_mean(torch.maximum(pg_loss1, pg_loss2), b_active[mb_inds])

                    value_error = new_values - b_returns[mb_inds]
                    if cfg.use_huber_loss:
                        value_loss = masked_mean(huber_loss(value_error, cfg.huber_delta), b_active[mb_inds])
                    else:
                        value_loss = 0.5 * masked_mean(value_error.pow(2), b_active[mb_inds])

                    entropy_loss = masked_mean(entropy, b_active[mb_inds])
                    loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    metrics = {
                        "loss": float(loss.item()),
                        "policy_loss": float(pg_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                    }
            return metrics

        metrics = {}
        for agent_id, policy in enumerate(self.policies):
            b_obs = buffer.obs[:, :, agent_id].reshape(-1, buffer.obs_dim)
            b_actions = buffer.actions[:, :, agent_id].reshape(-1)
            b_logprobs = buffer.logprobs[:, :, agent_id].reshape(-1).detach()
            b_returns = buffer.returns[:, :, agent_id].reshape(-1).detach()
            b_advantages = buffer.advantages[:, :, agent_id].reshape(-1).detach()
            b_active = buffer.active_masks[:, :, agent_id].reshape(-1)
            b_available = buffer.available_actions[:, :, agent_id].reshape(-1, buffer.action_dim)

            active_count = b_active.sum().clamp_min(1.0)
            adv_mean = (b_advantages * b_active).sum() / active_count
            adv_var = ((b_advantages - adv_mean).pow(2) * b_active).sum() / active_count
            b_advantages = (b_advantages - adv_mean) / torch.sqrt(adv_var + 1e-8)

            batch_size = b_obs.shape[0]
            minibatch_size = min(cfg.minibatch_size, batch_size)
            inds = np.arange(batch_size)
            for _ in range(cfg.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = torch.as_tensor(inds[start : start + minibatch_size], dtype=torch.long, device=self.device)
                    dist = policy.distribution(b_obs[mb_inds], b_available[mb_inds])
                    new_logprobs = dist.log_prob(b_actions[mb_inds])
                    entropy = dist.entropy()
                    new_values = policy.value(b_obs[mb_inds])

                    logratio = new_logprobs - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb_inds]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    pg_loss = masked_mean(torch.maximum(pg_loss1, pg_loss2), b_active[mb_inds])

                    value_error = new_values - b_returns[mb_inds]
                    if cfg.use_huber_loss:
                        value_loss = masked_mean(huber_loss(value_error, cfg.huber_delta), b_active[mb_inds])
                    else:
                        value_loss = 0.5 * masked_mean(value_error.pow(2), b_active[mb_inds])

                    entropy_loss = masked_mean(entropy, b_active[mb_inds])
                    loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                    self.optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    self.optimizers[agent_id].step()

                    metrics = {
                        "loss": float(loss.item()),
                        "policy_loss": float(pg_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                    }
        return metrics

    def save(self, path: str | Path) -> None:
        if self.share_parameters:
            payload = {
                "share_parameters": True,
                "policy": self.shared_policy.state_dict(),
                "optimizer": self.optimizers[0].state_dict(),
                "config": self.cfg.__dict__,
            }
        else:
            payload = {
                "share_parameters": False,
                "policies": [policy.state_dict() for policy in self.policies],
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "config": self.cfg.__dict__,
            }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if self.share_parameters:
            state = checkpoint.get("policy")
            if state is None:
                state = checkpoint["policies"][0]
            self.shared_policy.load_state_dict(state)
            if "optimizer" in checkpoint:
                self.optimizers[0].load_state_dict(checkpoint["optimizer"])
        else:
            for policy, state in zip(self.policies, checkpoint["policies"]):
                policy.load_state_dict(state)
            for optimizer, state in zip(self.optimizers, checkpoint.get("optimizers", [])):
                optimizer.load_state_dict(state)
