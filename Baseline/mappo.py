from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def infer_local_obs_layout(obs_dim: int, map_channels: int = 3, vector_dim: int = 3) -> Tuple[int, int, int]:
    image_values = obs_dim - vector_dim
    if image_values <= 0 or image_values % map_channels != 0:
        raise ValueError(
            f"obs_dim={obs_dim} cannot be split into {map_channels} map channels and {vector_dim} vector values."
        )
    local_cells = image_values // map_channels
    fov_size = int(np.sqrt(local_cells))
    if fov_size * fov_size != local_cells:
        raise ValueError(f"obs_dim={obs_dim} does not contain square local maps.")
    return map_channels, fov_size, vector_dim


@dataclass
class MAPPOConfig:
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
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    use_clipped_value_loss: bool = True
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    use_policy_active_masks: bool = True
    use_value_active_masks: bool = True
    use_proper_time_limits: bool = True
    use_valuenorm: bool = True
    value_norm_beta: float = 0.99999
    value_norm_epsilon: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_error = error.abs()
    quadratic = torch.minimum(abs_error, torch.tensor(delta, dtype=error.dtype, device=error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def masked_mean(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.to(dtype=values.dtype, device=values.device)
    return (values * masks).sum() / masks.sum().clamp_min(1.0)


def apply_available_actions_mask(
    logits: torch.Tensor,
    available_actions: Optional[torch.Tensor],
) -> torch.Tensor:
    if available_actions is None:
        return logits

    available = available_actions.to(device=logits.device) > 0.5
    if available.dim() == 1:
        available = available.unsqueeze(0)
    if available.shape != logits.shape:
        raise ValueError(
            f"available_actions shape {tuple(available.shape)} does not match logits shape {tuple(logits.shape)}."
        )

    no_available = ~available.any(dim=-1, keepdim=True)
    if no_available.any():
        available = torch.where(no_available, torch.ones_like(available), available)
    return logits.masked_fill(~available, -1.0e10)


class ValueNorm(nn.Module):
    """Running value target normalizer, following the MAPPO ValueNorm idea."""

    def __init__(
        self,
        shape: int | Tuple[int, ...] = 1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(self.shape, dtype=torch.float32, device=device))
        self.register_buffer("running_mean_sq", torch.zeros(self.shape, dtype=torch.float32, device=device))
        self.register_buffer("debiasing_term", torch.zeros(self.shape, dtype=torch.float32, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device=self.running_mean.device, dtype=torch.float32)
        if values.shape[-len(self.shape) :] != self.shape:
            values = values.reshape(-1, *self.shape)
        reduce_dims = tuple(range(values.dim() - len(self.shape)))
        batch_mean = values.mean(dim=reduce_dims)
        batch_mean_sq = values.pow(2).mean(dim=reduce_dims)

        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    @property
    def mean(self) -> torch.Tensor:
        initialized = self.debiasing_term > self.epsilon
        debiased_mean = self.running_mean / self.debiasing_term.clamp_min(self.epsilon)
        return torch.where(initialized, debiased_mean, torch.zeros_like(debiased_mean))

    @property
    def var(self) -> torch.Tensor:
        mean = self.mean
        initialized = self.debiasing_term > self.epsilon
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp_min(self.epsilon)
        mean_sq = torch.where(initialized, debiased_mean_sq, torch.ones_like(debiased_mean_sq))
        return (mean_sq - mean.pow(2)).clamp_min(self.epsilon)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / torch.sqrt(self.var.to(values.device))

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * torch.sqrt(self.var.to(values.device)) + self.mean.to(values.device)


class ActorFeatureEncoder(nn.Module):
    """Enhanced_MAPPO-style local spatial encoder for actor decisions."""

    def __init__(
        self,
        obs_dim: int,
        map_repr_dim: int = 256,
        goal_repr_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.map_channels, self.fov_size, self.vector_dim = infer_local_obs_layout(obs_dim)
        self.map_values = self.map_channels * self.fov_size * self.fov_size
        self.map_repr_dim = map_repr_dim
        self.goal_repr_dim = goal_repr_dim
        self.hidden_dim = hidden_dim

        final_spatial_size = self.fov_size // 4
        if final_spatial_size < 1:
            raise ValueError(f"fov_size is too small: {self.fov_size}")

        self.map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(self.map_channels, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            layer_init(nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            layer_init(nn.Conv2d(256, self.map_repr_dim, kernel_size=final_spatial_size)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(self.vector_dim, self.goal_repr_dim)),
            nn.ReLU(),
        )
        fusion_dim = self.map_repr_dim + self.goal_repr_dim
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        self.residual_fc = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim), std=1.0),
        )
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs shape (batch, {self.obs_dim}), got {tuple(obs.shape)}.")
        local_maps = obs[:, : self.map_values].reshape(
            -1,
            self.map_channels,
            self.fov_size,
            self.fov_size,
        )
        goal_vector = obs[:, self.map_values : self.map_values + self.vector_dim]
        map_features = self.map_encoder(local_maps)
        goal_features = self.goal_encoder(goal_vector)
        fused_features = torch.cat([map_features, goal_features], dim=-1)
        hidden = self.fusion(fused_features)
        residual = self.residual_fc(hidden)
        return self.activation(hidden + residual)


class CriticFeatureEncoder(nn.Module):
    """Enhanced_MAPPO-style per-agent encoder for the centralized critic."""

    def __init__(
        self,
        obs_dim: int,
        map_repr_dim: int = 128,
        goal_repr_dim: int = 16,
        agent_feature_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.map_channels, self.fov_size, self.vector_dim = infer_local_obs_layout(obs_dim)
        self.map_values = self.map_channels * self.fov_size * self.fov_size
        self.map_repr_dim = map_repr_dim
        self.goal_repr_dim = goal_repr_dim
        self.agent_feature_dim = agent_feature_dim

        final_spatial_size = self.fov_size // 4
        if final_spatial_size < 1:
            raise ValueError(f"fov_size is too small: {self.fov_size}")

        self.map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            layer_init(nn.Conv2d(128, self.map_repr_dim, kernel_size=final_spatial_size)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(self.vector_dim, self.goal_repr_dim)),
            nn.ReLU(),
        )
        fusion_dim = self.map_repr_dim + self.goal_repr_dim
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, self.agent_feature_dim)),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs shape (batch, {self.obs_dim}), got {tuple(obs.shape)}.")
        local_maps = obs[:, : self.map_values].reshape(
            -1,
            self.map_channels,
            self.fov_size,
            self.fov_size,
        )
        goal_vector = obs[:, self.map_values : self.map_values + self.vector_dim]
        map_features = self.map_encoder(local_maps)
        goal_features = self.goal_encoder(goal_vector)
        fused_features = torch.cat([map_features, goal_features], dim=-1)
        return self.fusion(fused_features)


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = ActorFeatureEncoder(
            obs_dim=obs_dim,
            map_repr_dim=256,
            goal_repr_dim=32,
            hidden_dim=hidden_dim,
        )
        self.policy_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.policy_head(self.encoder(obs))
        return apply_available_actions_mask(logits, available_actions)

    def distribution(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> Categorical:
        return Categorical(logits=self.forward(obs, available_actions))


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        global_obs_dim: int,
        num_agents: int,
        hidden_dim: int,
        agent_feature_dim: int = 128,
    ):
        super().__init__()
        if global_obs_dim % num_agents != 0:
            raise ValueError("global_obs_dim must be divisible by num_agents.")
        self.global_obs_dim = global_obs_dim
        self.num_agents = num_agents
        self.obs_dim = global_obs_dim // num_agents
        self.agent_feature_dim = agent_feature_dim
        self.encoder = CriticFeatureEncoder(
            obs_dim=self.obs_dim,
            map_repr_dim=128,
            goal_repr_dim=16,
            agent_feature_dim=self.agent_feature_dim,
        )
        critic_input_dim = self.agent_feature_dim * 3
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim // 2, 1), std=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
        if global_obs.dim() != 2 or global_obs.shape[-1] != self.global_obs_dim:
            raise ValueError(f"Expected global_obs shape (batch, {self.global_obs_dim}), got {tuple(global_obs.shape)}.")
        batch_size = global_obs.shape[0]
        per_agent_obs = global_obs.reshape(batch_size, self.num_agents, self.obs_dim)
        per_agent_obs_flat = per_agent_obs.reshape(batch_size * self.num_agents, self.obs_dim)
        per_agent_features = self.encoder(per_agent_obs_flat)
        per_agent_features = per_agent_features.reshape(
            batch_size,
            self.num_agents,
            self.agent_feature_dim,
        )

        mean_context = per_agent_features.mean(dim=1, keepdim=True)
        max_context = per_agent_features.max(dim=1, keepdim=True).values
        mean_context = mean_context.expand(-1, self.num_agents, -1)
        max_context = max_context.expand(-1, self.num_agents, -1)

        critic_input = torch.cat([per_agent_features, mean_context, max_context], dim=-1)
        critic_input = critic_input.reshape(batch_size * self.num_agents, -1)
        return self.value_head(critic_input).reshape(batch_size, self.num_agents)


class RolloutBuffer:
    def __init__(
        self,
        steps: int,
        num_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        device: str,
        num_envs: int = 1,
    ):
        self.steps = steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((steps, num_envs, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.global_obs = torch.zeros((steps, num_envs, global_obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((steps, num_envs, num_agents), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.bad_masks = torch.ones((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.active_masks = torch.ones((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.available_actions = None
        self.values = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_envs, num_agents), dtype=torch.float32, device=device)
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
        active_masks: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        bad_masks: Optional[torch.Tensor] = None,
    ) -> None:
        if self.ptr >= self.steps:
            raise RuntimeError("RolloutBuffer is full.")
        if available_actions is not None and self.available_actions is None:
            action_dim = int(available_actions.shape[-1])
            self.available_actions = torch.ones(
                (self.steps, self.num_envs, self.num_agents, action_dim),
                dtype=torch.float32,
                device=self.device,
            )
        self.obs[self.ptr].copy_(obs)
        self.global_obs[self.ptr].copy_(global_obs)
        self.actions[self.ptr].copy_(actions)
        self.logprobs[self.ptr].copy_(logprobs)
        self.rewards[self.ptr].copy_(rewards)
        self.dones[self.ptr].copy_(dones)
        self.values[self.ptr].copy_(values)
        self.active_masks[self.ptr].copy_(active_masks)
        if bad_masks is not None:
            self.bad_masks[self.ptr].copy_(bad_masks)
        if available_actions is not None:
            self.available_actions[self.ptr].copy_(available_actions)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True,
        value_normalizer: Optional[ValueNorm] = None,
    ) -> None:
        last_gae = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            if value_normalizer is not None:
                next_values_raw = value_normalizer.denormalize(next_values)
                current_values_raw = value_normalizer.denormalize(self.values[t])
            else:
                next_values_raw = next_values
                current_values_raw = self.values[t]

            delta = self.rewards[t] + gamma * next_values_raw * next_non_terminal - current_values_raw
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            if use_proper_time_limits:
                last_gae = last_gae * self.bad_masks[t]
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + current_values_raw


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
        self.value_normalizer = (
            ValueNorm(
                1,
                beta=config.value_norm_beta,
                epsilon=config.value_norm_epsilon,
                device=self.device,
            )
            if config.use_valuenorm
            else None
        )
        actor_lr = config.actor_lr if config.actor_lr is not None else config.learning_rate
        critic_lr = config.critic_lr if config.critic_lr is not None else config.learning_rate
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.distribution(obs, available_actions)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        if global_obs.dim() == 1:
            values = self.critic(global_obs.unsqueeze(0)).squeeze(0)
        elif global_obs.dim() == 2:
            values = self.critic(global_obs)
        else:
            raise ValueError(f"Expected global_obs dim 1 or 2, got shape {tuple(global_obs.shape)}.")
        return actions, logprobs, values

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.actor(obs, available_actions)
        return torch.argmax(logits, dim=-1)

    def _value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        active_masks: torch.Tensor,
    ) -> torch.Tensor:
        value_pred_clipped = old_values + (values - old_values).clamp(
            -self.cfg.clip_coef,
            self.cfg.clip_coef,
        )
        if self.value_normalizer is not None:
            with torch.no_grad():
                valid_returns = returns[active_masks > 0.5]
                self.value_normalizer.update(valid_returns if valid_returns.numel() > 0 else returns)
            value_targets = self.value_normalizer.normalize(returns)
        else:
            value_targets = returns

        error_original = value_targets - values
        error_clipped = value_targets - value_pred_clipped

        if self.cfg.use_huber_loss:
            value_loss_original = huber_loss(error_original, self.cfg.huber_delta)
            value_loss_clipped = huber_loss(error_clipped, self.cfg.huber_delta)
        else:
            value_loss_original = 0.5 * error_original.pow(2)
            value_loss_clipped = 0.5 * error_clipped.pow(2)

        if self.cfg.use_clipped_value_loss:
            value_loss = torch.maximum(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.cfg.use_value_active_masks:
            return masked_mean(value_loss, active_masks)
        return value_loss.mean()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        steps = buffer.steps
        envs = buffer.num_envs
        n = self.num_agents
        total_samples = steps * envs * n

        b_obs = buffer.obs.reshape(total_samples, self.obs_dim)
        b_global_unique = buffer.global_obs.reshape(steps * envs, self.global_obs_dim)
        b_global_ids = torch.arange(steps * envs, device=self.device).repeat_interleave(n)
        b_actions = buffer.actions.reshape(total_samples)
        b_logprobs = buffer.logprobs.reshape(total_samples)
        b_advantages = buffer.advantages.reshape(total_samples)
        b_returns = buffer.returns.reshape(total_samples)
        b_old_values = buffer.values.reshape(total_samples)
        b_active_masks = buffer.active_masks.reshape(total_samples)
        b_available_actions = (
            buffer.available_actions.reshape(total_samples, self.action_dim)
            if buffer.available_actions is not None
            else None
        )
        b_agent_ids = torch.arange(n, device=self.device).repeat(steps * envs)

        if self.cfg.use_policy_active_masks:
            valid_advantages = b_advantages[b_active_masks > 0.5]
            if valid_advantages.numel() > 1:
                adv_mean = valid_advantages.mean()
                adv_std = valid_advantages.std()
            else:
                adv_mean = b_advantages.mean()
                adv_std = b_advantages.std()
        else:
            adv_mean = b_advantages.mean()
            adv_std = b_advantages.std()
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)
        batch_size = total_samples
        minibatch_size = min(self.cfg.minibatch_size, batch_size)

        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }
        update_count = 0
        for _ in range(self.cfg.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_available_actions = b_available_actions[mb_idx] if b_available_actions is not None else None
                mb_active_masks = b_active_masks[mb_idx]

                dist = self.actor.distribution(b_obs[mb_idx], mb_available_actions)
                new_logprob = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()

                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                policy_loss_items = torch.max(pg_loss1, pg_loss2)
                if self.cfg.use_policy_active_masks:
                    policy_loss = masked_mean(policy_loss_items, mb_active_masks)
                else:
                    policy_loss = policy_loss_items.mean()

                self.actor_optimizer.zero_grad()
                actor_loss = policy_loss - self.cfg.ent_coef * entropy
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()

                unique_global_ids, inverse_global_ids = torch.unique(
                    b_global_ids[mb_idx],
                    sorted=False,
                    return_inverse=True,
                )
                values_all = self.critic(b_global_unique[unique_global_ids])
                values = values_all[inverse_global_ids, b_agent_ids[mb_idx]]
                value_loss = self._value_loss(
                    values,
                    b_old_values[mb_idx],
                    b_returns[mb_idx],
                    mb_active_masks,
                )

                self.critic_optimizer.zero_grad()
                critic_loss = self.cfg.vf_coef * value_loss
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

                loss = actor_loss + critic_loss
                metrics["loss"] += float(loss.item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["actor_grad_norm"] += float(actor_grad_norm.item())
                metrics["critic_grad_norm"] += float(critic_grad_norm.item())
                metrics["ratio"] += float(ratio.mean().item())
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
                "value_normalizer": self.value_normalizer.state_dict()
                if self.value_normalizer is not None
                else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.value_normalizer is not None and checkpoint.get("value_normalizer") is not None:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
