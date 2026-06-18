from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from emappo_env import MultiAgentGridEvacuationEnv


class CloudpickleWrapper:
    def __init__(self, fn: Callable[[], MultiAgentGridEvacuationEnv]):
        self.fn = fn

    def __getstate__(self):
        import pickle

        return pickle.dumps(self.fn)

    def __setstate__(self, state):
        import pickle

        self.fn = pickle.loads(state)


@dataclass
class EnvView:
    n_agents: int
    obs_dim: int
    n_actions: int
    rows: int
    cols: int
    grid_map: np.ndarray
    starts: np.ndarray
    exits: np.ndarray


def _env_metadata(env: MultiAgentGridEvacuationEnv) -> dict:
    return {
        "n_agents": env.n_agents,
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "rows": env.rows,
        "cols": env.cols,
        "grid_map": env.grid_map.copy(),
        "starts": env.starts.copy(),
        "exits": env.exits.copy(),
    }


def _paths_copy(env: MultiAgentGridEvacuationEnv):
    return [[point.copy() for point in path] for path in env.paths]


def _worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:
    parent_remote.close()
    env = env_fn_wrapper.fn()
    try:
        while True:
            command, data = remote.recv()
            if command == "step":
                remote.send(env.step(data))
            elif command == "reset":
                remote.send(env.reset(seed=data))
            elif command == "available_actions":
                remote.send(env.available_actions())
            elif command == "get_paths":
                remote.send(_paths_copy(env))
            elif command == "get_metadata":
                remote.send(_env_metadata(env))
            elif command == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(command)
    except KeyboardInterrupt:
        pass


class DummyVecEnv:
    """Synchronous vectorized environment wrapper.

    Multiple envs are stepped sequentially in the current Python process.
    Use SubprocVecEnv for real multi-process rollout collection.
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

    def get_paths(self, index: int = 0):
        return _paths_copy(self.envs[index])

    def close(self) -> None:
        return None


class SubprocVecEnv:
    """True multi-process vectorized environment wrapper."""

    def __init__(self, env_fns: Sequence[Callable[[], MultiAgentGridEvacuationEnv]]):
        if not env_fns:
            raise ValueError("SubprocVecEnv requires at least one environment.")
        self.waiting = False
        self.closed = False
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = ctx.Process(
                target=_worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        metadata = self._recv_from(0, "get_metadata")
        self.envs = [EnvView(**metadata) for _ in env_fns]
        self._n_agents = int(metadata["n_agents"])
        self._obs_dim = int(metadata["obs_dim"])
        self._n_actions = int(metadata["n_actions"])

    def _recv_from(self, index: int, command: str, data=None):
        self.remotes[index].send((command, data))
        return self.remotes[index].recv()

    @property
    def num_envs(self) -> int:
        return len(self.remotes)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        for idx, remote in enumerate(self.remotes):
            env_seed = None if seed is None else seed + idx
            remote.send(("reset", env_seed))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs, axis=0), list(infos)

    def reset_at(self, index: int, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        return self._recv_from(index, "reset", seed)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.n_agents):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.n_agents)}, got {tuple(actions.shape)}."
            )
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs, axis=0),
            np.stack(rewards, axis=0),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def available_actions(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("available_actions", None))
        return np.stack([remote.recv() for remote in self.remotes], axis=0)

    def get_paths(self, index: int = 0):
        return self._recv_from(index, "get_paths")

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


def make_vec_env(
    env_fns: Sequence[Callable[[], MultiAgentGridEvacuationEnv]],
    vec_env_type: str = "dummy",
):
    vec_env_type = (vec_env_type or "dummy").lower()
    if vec_env_type == "subproc" and len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    if vec_env_type not in {"dummy", "subproc"}:
        raise ValueError("vec_env_type must be 'dummy' or 'subproc'.")
    return DummyVecEnv(env_fns)
