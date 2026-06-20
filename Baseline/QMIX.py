from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mappo import layer_init


# This follows the PyMARL/QMIX structure as closely as this grid environment
# allows: recurrent shared agent network, previous-action + agent-id inputs,
# episode replay, hypernetwork mixer, double-Q target action selection.
@dataclass
class QMIXConfig:
    total_timesteps: int = 200_000
    buffer_size: int = 5000
    batch_size: int = 32
    train_start_episodes: int = 8
    train_interval_episodes: int = 1
    gradient_steps_per_episode: int = 4
    target_update_interval_episodes: int = 20
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    rnn_hidden_dim: int = 64
    mixing_embed_dim: int = 32
    hypernet_embed_dim: int = 64
    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RNNAgent(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = layer_init(nn.Linear(hidden_dim, action_dim), std=1.0)

    def init_hidden(self, batch_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_agents, self.hidden_dim, dtype=torch.float32, device=device)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    def __init__(self, num_agents: int, state_dim: int, mixing_dim: int, hypernet_dim: int):
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.mixing_dim = int(mixing_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, num_agents * mixing_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, mixing_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.mixing_dim, 1)
        v = self.value(states).view(batch_size, 1, 1)
        y = torch.bmm(hidden, w2) + v
        return y.view(batch_size)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.episodes: list[dict[str, np.ndarray]] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: dict[str, np.ndarray]) -> None:
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if len(self.episodes) < batch_size:
            raise ValueError("Not enough episodes in replay buffer.")
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=False)
        episodes = [self.episodes[int(idx)] for idx in indices]
        max_t = max(ep["actions"].shape[0] for ep in episodes)
        n_agents = episodes[0]["obs"].shape[1]
        obs_dim = episodes[0]["obs"].shape[2]
        state_dim = episodes[0]["states"].shape[1]
        action_dim = episodes[0]["avail_actions"].shape[2]

        obs = np.zeros((batch_size, max_t + 1, n_agents, obs_dim), dtype=np.float32)
        states = np.zeros((batch_size, max_t + 1, state_dim), dtype=np.float32)
        avail_actions = np.zeros((batch_size, max_t + 1, n_agents, action_dim), dtype=np.float32)
        actions = np.zeros((batch_size, max_t, n_agents), dtype=np.int64)
        rewards = np.zeros((batch_size, max_t), dtype=np.float32)
        terminated = np.zeros((batch_size, max_t), dtype=np.float32)
        filled = np.zeros((batch_size, max_t), dtype=np.float32)

        for batch_idx, episode in enumerate(episodes):
            t = episode["actions"].shape[0]
            obs[batch_idx, : t + 1] = episode["obs"]
            states[batch_idx, : t + 1] = episode["states"]
            avail_actions[batch_idx, : t + 1] = episode["avail_actions"]
            actions[batch_idx, :t] = episode["actions"]
            rewards[batch_idx, :t] = episode["rewards"]
            terminated[batch_idx, :t] = episode["terminated"]
            filled[batch_idx, :t] = 1.0

        return {
            "obs": torch.tensor(obs, dtype=torch.float32, device=device),
            "states": torch.tensor(states, dtype=torch.float32, device=device),
            "avail_actions": torch.tensor(avail_actions, dtype=torch.float32, device=device),
            "actions": torch.tensor(actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "terminated": torch.tensor(terminated, dtype=torch.float32, device=device),
            "filled": torch.tensor(filled, dtype=torch.float32, device=device),
        }


class QMIXAgent:
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, cfg: QMIXConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.state_dim = self.obs_dim * self.num_agents
        self.agent_input_dim = self.obs_dim + self.action_dim + self.num_agents

        self.agent = RNNAgent(self.agent_input_dim, action_dim, cfg.rnn_hidden_dim).to(self.device)
        self.target_agent = RNNAgent(self.agent_input_dim, action_dim, cfg.rnn_hidden_dim).to(self.device)
        self.mixer = QMixer(num_agents, self.state_dim, cfg.mixing_embed_dim, cfg.hypernet_embed_dim).to(self.device)
        self.target_mixer = QMixer(num_agents, self.state_dim, cfg.mixing_embed_dim, cfg.hypernet_embed_dim).to(self.device)
        self.optimizer = optim.RMSprop(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            lr=cfg.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )
        self.agent_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device)
        self.update_targets()

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def epsilon(self, global_step: int) -> float:
        frac = min(1.0, global_step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def _build_inputs(self, obs: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        agent_ids = self.agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([obs, prev_actions, agent_ids], dim=-1).reshape(batch_size * self.num_agents, -1)

    @torch.no_grad()
    def select_actions(
        self,
        obs: np.ndarray,
        avail_actions: np.ndarray,
        prev_actions: np.ndarray,
        hidden: torch.Tensor,
        epsilon: float,
    ) -> tuple[np.ndarray, torch.Tensor]:
        obs_tensor = torch.tensor(obs[None, ...], dtype=torch.float32, device=self.device)
        prev_tensor = torch.tensor(prev_actions[None, ...], dtype=torch.float32, device=self.device)
        inputs = self._build_inputs(obs_tensor, prev_tensor)
        q_values, next_hidden = self.agent(inputs, hidden)
        q_values = q_values.view(self.num_agents, self.action_dim).cpu().numpy()
        masked_q = np.where(avail_actions > 0.5, q_values, -1.0e9)
        actions = np.argmax(masked_q, axis=-1).astype(np.int64)
        for agent_id in range(self.num_agents):
            if random.random() < epsilon:
                valid = np.flatnonzero(avail_actions[agent_id] > 0.5)
                if len(valid) > 0:
                    actions[agent_id] = int(np.random.choice(valid))
        return actions, next_hidden

    def _forward_sequence(self, batch: dict[str, torch.Tensor], target: bool = False) -> torch.Tensor:
        obs = batch["obs"]
        actions = batch["actions"]
        batch_size, max_t_plus_one, _, _ = obs.shape
        max_t = max_t_plus_one - 1
        net = self.target_agent if target else self.agent
        hidden = net.init_hidden(batch_size * self.num_agents, self.device)
        outputs = []
        zero_prev = torch.zeros(batch_size, self.num_agents, self.action_dim, dtype=torch.float32, device=self.device)
        for t in range(max_t_plus_one):
            if t == 0:
                prev_actions = zero_prev
            else:
                prev_actions = nn.functional.one_hot(actions[:, t - 1], num_classes=self.action_dim).float()
            inputs = self._build_inputs(obs[:, t], prev_actions)
            q_values, hidden = net(inputs, hidden)
            outputs.append(q_values.view(batch_size, self.num_agents, self.action_dim))
        return torch.stack(outputs, dim=1)

    def train_step(self, replay: EpisodeReplayBuffer) -> dict:
        cfg = self.cfg
        batch = replay.sample(cfg.batch_size, self.device)
        batch_size, max_t = batch["actions"].shape[:2]
        mask = batch["filled"]

        mac_out = self._forward_sequence(batch, target=False)
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1],
            dim=3,
            index=batch["actions"].unsqueeze(-1),
        ).squeeze(-1)

        target_mac_out = self._forward_sequence(batch, target=True)[:, 1:]
        target_mac_out = target_mac_out.masked_fill(batch["avail_actions"][:, 1:] <= 0.5, -1.0e9)

        with torch.no_grad():
            live_mac_out = mac_out[:, 1:].detach().masked_fill(batch["avail_actions"][:, 1:] <= 0.5, -1.0e9)
            cur_max_actions = live_mac_out.max(dim=3, keepdim=True).indices
            target_max_qvals = torch.gather(target_mac_out, dim=3, index=cur_max_actions).squeeze(-1)

            target_q_tot = self.target_mixer(
                target_max_qvals.reshape(batch_size * max_t, self.num_agents),
                batch["states"][:, 1:].reshape(batch_size * max_t, self.state_dim),
            ).view(batch_size, max_t)
            targets = batch["rewards"] + cfg.gamma * (1.0 - batch["terminated"]) * target_q_tot

        chosen_q_tot = self.mixer(
            chosen_action_qvals.reshape(batch_size * max_t, self.num_agents),
            batch["states"][:, :-1].reshape(batch_size * max_t, self.state_dim),
        ).view(batch_size, max_t)

        td_error = chosen_q_tot - targets.detach()
        masked_td_error = td_error * mask
        loss = masked_td_error.pow(2).sum() / mask.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.mixer.parameters()), cfg.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "q_total": float((chosen_q_tot * mask).sum().item() / mask.sum().clamp_min(1.0).item()),
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_agent": self.target_agent.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )
