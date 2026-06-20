from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from mappo import (
    ActorFeatureEncoder,
    CriticFeatureEncoder,
    MAPPOConfig,
    ValueNorm,
    apply_available_actions_mask,
    huber_loss,
    layer_init,
    masked_mean,
)


class RNNLayer(nn.Module):
    """GRU layer following the official on-policy MAPPO recurrent design."""

    def __init__(self, input_dim: int, hidden_dim: int, recurrent_layers: int = 1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.recurrent_layers = int(recurrent_layers)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=self.recurrent_layers)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if hxs.dim() == 2:
            hxs = hxs.unsqueeze(1).repeat(1, self.recurrent_layers, 1)

        if x.size(0) == hxs.size(0):
            masked_hxs = (hxs * masks.view(-1, 1, 1)).transpose(0, 1).contiguous()
            x, hxs = self.rnn(x.unsqueeze(0), masked_hxs)
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            batch = hxs.size(0)
            time = int(x.size(0) / batch)
            x = x.view(time, batch, x.size(-1))
            masks = masks.view(time, batch)
            hxs = hxs.transpose(0, 1)
            outputs = []
            for step in range(time):
                hxs = hxs * masks[step].view(1, batch, 1)
                out, hxs = self.rnn(x[step : step + 1], hxs.contiguous())
                outputs.append(out)
            x = torch.cat(outputs, dim=0).reshape(time * batch, -1)
            hxs = hxs.transpose(0, 1)

        return self.norm(x), hxs


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, recurrent_layers: int = 1):
        super().__init__()
        self.encoder = ActorFeatureEncoder(
            obs_dim=obs_dim,
            map_repr_dim=256,
            goal_repr_dim=32,
            hidden_dim=hidden_dim,
        )
        self.rnn = RNNLayer(hidden_dim, hidden_dim, recurrent_layers)
        self.policy_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        features, rnn_states = self.rnn(features, rnn_states, masks)
        logits = self.policy_head(features)
        return apply_available_actions_mask(logits, available_actions), rnn_states

    def distribution(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        logits, rnn_states = self.forward(obs, rnn_states, masks, available_actions)
        return Categorical(logits=logits), rnn_states


class RecurrentCentralizedCritic(nn.Module):
    def __init__(
        self,
        global_obs_dim: int,
        num_agents: int,
        hidden_dim: int,
        recurrent_layers: int = 1,
        agent_feature_dim: int = 128,
    ):
        super().__init__()
        if global_obs_dim % num_agents != 0:
            raise ValueError("global_obs_dim must be divisible by num_agents.")
        self.global_obs_dim = int(global_obs_dim)
        self.num_agents = int(num_agents)
        self.obs_dim = self.global_obs_dim // self.num_agents
        self.agent_feature_dim = int(agent_feature_dim)
        self.encoder = CriticFeatureEncoder(
            obs_dim=self.obs_dim,
            map_repr_dim=128,
            goal_repr_dim=16,
            agent_feature_dim=self.agent_feature_dim,
        )
        self.pre_rnn = nn.Sequential(
            layer_init(nn.Linear(self.agent_feature_dim * 3, hidden_dim)),
            nn.ReLU(),
        )
        self.rnn = RNNLayer(hidden_dim, hidden_dim, recurrent_layers)
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim // 2, 1), std=1.0),
        )

    def forward(
        self,
        global_obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
        batch_size = global_obs.shape[0]
        per_agent_obs = global_obs.reshape(batch_size, self.num_agents, self.obs_dim)
        flat_obs = per_agent_obs.reshape(batch_size * self.num_agents, self.obs_dim)
        per_agent_features = self.encoder(flat_obs).reshape(batch_size, self.num_agents, self.agent_feature_dim)

        mean_context = per_agent_features.mean(dim=1, keepdim=True).expand(-1, self.num_agents, -1)
        max_context = per_agent_features.max(dim=1, keepdim=True).values.expand(-1, self.num_agents, -1)
        critic_input = torch.cat([per_agent_features, mean_context, max_context], dim=-1)
        critic_input = self.pre_rnn(critic_input.reshape(batch_size * self.num_agents, -1))

        flat_states = rnn_states.reshape(batch_size * self.num_agents, *rnn_states.shape[2:])
        flat_masks = masks.reshape(batch_size * self.num_agents, 1)
        features, new_states = self.rnn(critic_input, flat_states, flat_masks)
        values = self.value_head(features).reshape(batch_size, self.num_agents)
        new_states = new_states.reshape(batch_size, self.num_agents, *new_states.shape[1:])
        return values, new_states


class RecurrentRolloutBuffer:
    def __init__(
        self,
        steps: int,
        num_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        hidden_dim: int,
        device: str,
        num_envs: int = 1,
        recurrent_layers: int = 1,
    ):
        self.steps = steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((steps, num_envs, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.global_obs = torch.zeros((steps, num_envs, global_obs_dim), dtype=torch.float32, device=device)
        self.actor_rnn_states = torch.zeros(
            (steps, num_envs, num_agents, recurrent_layers, hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        self.critic_rnn_states = torch.zeros_like(self.actor_rnn_states)
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
        actor_rnn_states: torch.Tensor,
        critic_rnn_states: torch.Tensor,
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
        self.actor_rnn_states[self.ptr].copy_(actor_rnn_states)
        self.critic_rnn_states[self.ptr].copy_(critic_rnn_states)
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


@dataclass
class RMAPPOConfig(MAPPOConfig):
    recurrent_layers: int = 1


class RMAPPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, config: RMAPPOConfig):
        self.cfg = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_obs_dim = obs_dim * num_agents
        self.device = config.device
        self.hidden_dim = config.hidden_dim
        self.recurrent_layers = config.recurrent_layers

        self.actor = RecurrentActor(obs_dim, action_dim, config.hidden_dim, config.recurrent_layers).to(self.device)
        self.critic = RecurrentCentralizedCritic(
            self.global_obs_dim,
            num_agents,
            config.hidden_dim,
            config.recurrent_layers,
        ).to(self.device)
        self.value_normalizer = (
            ValueNorm(1, beta=config.value_norm_beta, epsilon=config.value_norm_epsilon, device=self.device)
            if config.use_valuenorm
            else None
        )
        actor_lr = config.actor_lr if config.actor_lr is not None else config.learning_rate
        critic_lr = config.critic_lr if config.critic_lr is not None else config.learning_rate
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def initial_states(self, num_envs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (num_envs, self.num_agents, self.recurrent_layers, self.hidden_dim)
        actor_states = torch.zeros(shape, dtype=torch.float32, device=self.device)
        critic_states = torch.zeros(shape, dtype=torch.float32, device=self.device)
        return actor_states, critic_states

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor,
        actor_rnn_states: torch.Tensor,
        critic_rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_obs = obs.reshape(-1, self.obs_dim)
        flat_actor_states = actor_rnn_states.reshape(-1, self.recurrent_layers, self.hidden_dim)
        flat_masks = masks.reshape(-1, 1)
        flat_available = available_actions.reshape(-1, self.action_dim) if available_actions is not None else None
        dist, new_actor_states = self.actor.distribution(flat_obs, flat_actor_states, flat_masks, flat_available)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        values, new_critic_states = self.critic(global_obs, critic_rnn_states, masks)
        return (
            actions.reshape(obs.shape[0], self.num_agents),
            logprobs.reshape(obs.shape[0], self.num_agents),
            values,
            new_actor_states.reshape_as(actor_rnn_states),
            new_critic_states,
        )

    def _value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        active_masks: torch.Tensor,
    ) -> torch.Tensor:
        value_pred_clipped = old_values + (values - old_values).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
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
        value_loss = torch.maximum(value_loss_original, value_loss_clipped) if self.cfg.use_clipped_value_loss else value_loss_original
        if self.cfg.use_value_active_masks:
            return masked_mean(value_loss, active_masks)
        return value_loss.mean()

    def update(self, buffer: RecurrentRolloutBuffer) -> Dict[str, float]:
        steps, envs, n = buffer.steps, buffer.num_envs, self.num_agents
        total_samples = steps * envs * n
        b_obs = buffer.obs.reshape(total_samples, self.obs_dim)
        b_actor_states = buffer.actor_rnn_states.reshape(total_samples, self.recurrent_layers, self.hidden_dim)
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

        b_global_unique = buffer.global_obs.reshape(steps * envs, self.global_obs_dim)
        b_critic_states_unique = buffer.critic_rnn_states.reshape(
            steps * envs,
            n,
            self.recurrent_layers,
            self.hidden_dim,
        )
        b_global_masks_unique = buffer.active_masks.reshape(steps * envs, n)
        b_global_ids = torch.arange(steps * envs, device=self.device).repeat_interleave(n)
        b_agent_ids = torch.arange(n, device=self.device).repeat(steps * envs)

        valid_advantages = b_advantages[b_active_masks > 0.5] if self.cfg.use_policy_active_masks else b_advantages
        if valid_advantages.numel() > 1:
            b_advantages = (b_advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        else:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        minibatch_size = min(self.cfg.minibatch_size, total_samples)
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
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_available = b_available_actions[mb_idx] if b_available_actions is not None else None
                mb_active = b_active_masks[mb_idx]
                dist, _ = self.actor.distribution(
                    b_obs[mb_idx],
                    b_actor_states[mb_idx],
                    mb_active.unsqueeze(-1),
                    mb_available,
                )
                new_logprob = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()
                ratio = (new_logprob - b_logprobs[mb_idx]).exp()
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                policy_loss_items = torch.maximum(pg_loss1, pg_loss2)
                policy_loss = masked_mean(policy_loss_items, mb_active) if self.cfg.use_policy_active_masks else policy_loss_items.mean()

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
                values_all, _ = self.critic(
                    b_global_unique[unique_global_ids],
                    b_critic_states_unique[unique_global_ids],
                    b_global_masks_unique[unique_global_ids],
                )
                values = values_all[inverse_global_ids, b_agent_ids[mb_idx]]
                value_loss = self._value_loss(values, b_old_values[mb_idx], b_returns[mb_idx], mb_active)

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
                "hidden_dim": self.hidden_dim,
                "recurrent_layers": self.recurrent_layers,
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
