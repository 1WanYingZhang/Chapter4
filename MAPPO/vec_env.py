from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from mappo_env import MultiAgentGridEvacuationEnv


class DummyVecEnv:
    """Synchronous vectorized environment wrapper.

    This follows the light_mappo-style DummyVecEnv idea: multiple independent
    env instances are stepped through one Python interface. It improves data
    diversity per update and keeps the code ready for later Subproc/Ray rollout.
    """

    def __init__(self, env_fns: Sequence[Callable[[], MultiAgentGridEvacuationEnv]]):
        if not env_fns:
            raise ValueError("DummyVecEnv requires at least one environment.")
        self.envs: List[MultiAgentGridEvacuationEnv] = [fn() for fn in env_fns]
        first = self.envs[0]
        for env in self.envs[1:]:
            if env.n_agents != first.n_agents:
                raise ValueError("All vectorized envs must have the same number of agents.")
            if env.obs_dim != first.obs_dim:
                raise ValueError("All vectorized envs must have the same obs_dim.")
            if env.n_actions != first.n_actions:
                raise ValueError("All vectorized envs must have the same action space.")

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def n_agents(self) -> int:
        return self.envs[0].n_agents

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def n_actions(self) -> int:
        return self.envs[0].n_actions

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        obs_list = []
        infos = []
        for idx, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + idx
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list, axis=0), infos

    def reset_at(self, index: int, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        return self.envs[index].reset(seed=seed)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.n_agents):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.n_agents)}, got {tuple(actions.shape)}."
            )

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        infos = []
        for idx, env in enumerate(self.envs):
            obs, rewards, terminated, truncated, info = env.step(actions[idx])
            obs_list.append(obs)
            reward_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.stack(reward_list, axis=0),
            np.asarray(terminated_list, dtype=bool),
            np.asarray(truncated_list, dtype=bool),
            infos,
        )

    def available_actions(self) -> np.ndarray:
        return np.stack([env.available_actions() for env in self.envs], axis=0)
