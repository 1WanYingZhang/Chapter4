from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Allows syntax checks and light environment use before installing deps.
    class _FallbackEnv:
        pass

    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _MultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.shape = self.nvec.shape

        def sample(self):
            return np.array([np.random.randint(n) for n in self.nvec], dtype=np.int64)

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape) if shape is not None else np.shape(low)
            self.dtype = dtype

    class _Spaces:
        Box = _Box
        Discrete = _Discrete
        MultiDiscrete = _MultiDiscrete

    class _Gym:
        Env = _FallbackEnv

    gym = _Gym()
    spaces = _Spaces()


GridPoint = Tuple[int, int]


@dataclass
class EvacuationConfig:
    max_steps: int = 300
    local_view_radius: int = 2
    max_neighbors: int = 4
    allow_wait: bool = True
    exit_capacity: int = 1
    step_penalty: float = -0.05
    wait_penalty: float = -0.03
    obstacle_penalty: float = -1.0
    collision_penalty: float = -2.0
    swap_penalty: float = -2.0
    crowd_penalty: float = -0.15
    revisit_penalty: float = -0.03
    distance_reward: float = 1.0
    goal_reward: float = 10.0
    team_escape_reward: float = 0.25


class MultiAgentGridEvacuationEnv(gym.Env):
    """Discrete grid evacuation environment for MAPPO.

    Map convention is compatible with the existing single-agent Env.py:
    0 means free cell and 1 means obstacle. Agents move synchronously.
    """

    metadata = {"render_modes": ["ansi"]}

    BASE_MOVES = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
        ],
        dtype=np.int32,
    )

    WAIT_MOVE = np.array([[0, 0]], dtype=np.int32)

    def __init__(
        self,
        map_data: np.ndarray,
        starts: Sequence[GridPoint],
        exits: Sequence[GridPoint],
        config: Optional[EvacuationConfig] = None,
    ):
        super().__init__()
        self.cfg = config or EvacuationConfig()
        self.grid_map = np.asarray(map_data, dtype=np.int8).copy()
        if self.grid_map.ndim != 2:
            raise ValueError("map_data must be a 2D array.")

        self.rows, self.cols = self.grid_map.shape
        self.starts = np.asarray(starts, dtype=np.int32)
        self.exits = np.asarray(exits, dtype=np.int32)
        if self.starts.ndim != 2 or self.starts.shape[1] != 2:
            raise ValueError("starts must be a sequence of (row, col) points.")
        if self.exits.ndim != 2 or self.exits.shape[1] != 2:
            raise ValueError("exits must be a sequence of (row, col) points.")
        if len(self.exits) == 0:
            raise ValueError("At least one exit is required.")
        self.n_agents = int(len(self.starts))
        if len({tuple(point) for point in self.starts}) != self.n_agents:
            raise ValueError("Agent starts must be unique.")


        self.moves = (
            np.vstack([self.BASE_MOVES, self.WAIT_MOVE])
            if self.cfg.allow_wait
            else self.BASE_MOVES.copy()
        )
        self.n_actions = int(len(self.moves))

        self._validate_points(self.starts, "start")
        self._validate_points(self.exits, "exit")

        self.local_cells = (2 * self.cfg.local_view_radius + 1) ** 2
        self.obs_dim = 7 + self.local_cells + 3 * self.cfg.max_neighbors
        self.action_space = spaces.MultiDiscrete(np.full(self.n_agents, self.n_actions))
        self.single_action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )

        self.agent_pos = self.starts.copy()
        self.active = np.ones(self.n_agents, dtype=bool)
        self.arrival_steps = np.full(self.n_agents, -1, dtype=np.int32)
        self.prev_actions = np.full(self.n_agents, -1, dtype=np.int32)
        self.visited: List[set[GridPoint]] = []
        self.paths: List[List[np.ndarray]] = []
        self.steps = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = self.starts.copy()
        self.active = np.ones(self.n_agents, dtype=bool)
        self.arrival_steps = np.full(self.n_agents, -1, dtype=np.int32)
        self.prev_actions = np.full(self.n_agents, -1, dtype=np.int32)
        self.visited = [{tuple(pos)} for pos in self.agent_pos]
        self.paths = [[pos.copy()] for pos in self.agent_pos]
        self.steps = 0
        return self._get_obs(), self._info(escaped_this_step=0)

    def step(self, actions: Sequence[int]):
        self.steps += 1
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}.")

        prev_pos = self.agent_pos.copy()
        proposals = prev_pos.copy()
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        components = [self._empty_components() for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            if not self.active[i]:
                continue

            action = int(actions[i])
            if action < 0 or action >= self.n_actions:
                action = self.n_actions - 1 if self.cfg.allow_wait else 0
            self.prev_actions[i] = action
            move = self.moves[action]
            target = prev_pos[i] + move

            rewards[i] += self.cfg.step_penalty
            components[i]["step"] += self.cfg.step_penalty
            if np.array_equal(move, [0, 0]):
                rewards[i] += self.cfg.wait_penalty
                components[i]["wait"] += self.cfg.wait_penalty

            if self._blocked(target):
                proposals[i] = prev_pos[i]
                rewards[i] += self.cfg.obstacle_penalty
                components[i]["obstacle"] += self.cfg.obstacle_penalty
            else:
                proposals[i] = target

        proposals, failed_agents, swap_agents = self._resolve_motion_conflicts(prev_pos, proposals)
        for i in failed_agents:
            if not self.active[i]:
                continue
            penalty = self.cfg.collision_penalty
            if i in swap_agents:
                penalty += self.cfg.swap_penalty
                components[i]["swap"] += self.cfg.swap_penalty
            rewards[i] += penalty
            components[i]["collision"] += penalty

        escaped_this_step = 0
        for i in range(self.n_agents):
            if not self.active[i]:
                self.paths[i].append(self.agent_pos[i].copy())
                continue

            old_dist = self._distance_to_nearest_exit(prev_pos[i])
            new_dist = self._distance_to_nearest_exit(proposals[i])
            distance_delta = (old_dist - new_dist) / max(1.0, self._diagonal)
            dist_reward = self.cfg.distance_reward * distance_delta
            rewards[i] += dist_reward
            components[i]["distance"] += dist_reward

            new_key = tuple(int(x) for x in proposals[i])
            if new_key in self.visited[i] and not self._is_exit(proposals[i]):
                rewards[i] += self.cfg.revisit_penalty
                components[i]["revisit"] += self.cfg.revisit_penalty
            self.visited[i].add(new_key)

            crowd = self._local_crowd(i, proposals)
            if crowd > 0:
                crowd_penalty = self.cfg.crowd_penalty * crowd
                rewards[i] += crowd_penalty
                components[i]["crowd"] += crowd_penalty

            self.agent_pos[i] = proposals[i]
            if self._is_exit(self.agent_pos[i]):
                self.active[i] = False
                self.arrival_steps[i] = self.steps
                rewards[i] += self.cfg.goal_reward
                components[i]["goal"] += self.cfg.goal_reward
                escaped_this_step += 1

            self.paths[i].append(self.agent_pos[i].copy())

        if escaped_this_step > 0:
            team_bonus = self.cfg.team_escape_reward * escaped_this_step
            rewards += team_bonus
            for item in components:
                item["team"] += team_bonus

        terminated = bool(np.all(~self.active))
        truncated = bool(self.steps >= self.cfg.max_steps)
        info = self._info(escaped_this_step=escaped_this_step)
        info["reward_components"] = components
        return self._get_obs(), rewards, terminated, truncated, info

    @property
    def _diagonal(self) -> float:
        return float(np.hypot(self.rows - 1, self.cols - 1))

    def _validate_points(self, points: np.ndarray, name: str) -> None:
        for point in points:
            if self._out_of_bounds(point):
                raise ValueError(f"{name} point {tuple(point)} is outside the map.")
            if self.grid_map[point[0], point[1]] == 1:
                raise ValueError(f"{name} point {tuple(point)} is on an obstacle.")

    def _out_of_bounds(self, point: np.ndarray) -> bool:
        r, c = int(point[0]), int(point[1])
        return r < 0 or r >= self.rows or c < 0 or c >= self.cols

    def _blocked(self, point: np.ndarray) -> bool:
        if self._out_of_bounds(point):
            return True
        return bool(self.grid_map[int(point[0]), int(point[1])] == 1)

    def _is_exit(self, point: np.ndarray) -> bool:
        return any(np.array_equal(point, exit_point) for exit_point in self.exits)

    def _distance_to_nearest_exit(self, point: np.ndarray) -> float:
        deltas = self.exits - point
        return float(np.min(np.linalg.norm(deltas, axis=1)))

    def _nearest_exit_delta(self, point: np.ndarray) -> np.ndarray:
        deltas = self.exits - point
        idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
        return deltas[idx].astype(np.float32)

    def _resolve_motion_conflicts(
        self, prev_pos: np.ndarray, proposals: np.ndarray
    ) -> tuple[np.ndarray, set[int], set[int]]:
        final = proposals.copy()
        failed: set[int] = set()
        swap_failed: set[int] = set()

        for _ in range(self.n_agents + 1):
            new_failed: set[int] = set()
            vertex_failed = self._resolve_vertex_conflicts(final)
            swap_failed_now = self._resolve_swap_conflicts(prev_pos, final)
            occupied_failed = self._resolve_occupied_target_conflicts(prev_pos, final)

            new_failed.update(vertex_failed)
            new_failed.update(swap_failed_now)
            new_failed.update(occupied_failed)
            swap_failed.update(swap_failed_now)
            new_failed.difference_update(failed)

            if not new_failed:
                break

            for i in new_failed:
                final[i] = prev_pos[i]
            failed.update(new_failed)

        return final, failed, swap_failed

    def _resolve_vertex_conflicts(self, proposals: np.ndarray) -> set[int]:
        conflicts: set[int] = set()
        buckets: Dict[GridPoint, List[int]] = {}
        for i, point in enumerate(proposals):
            if not self.active[i]:
                continue
            key = tuple(int(x) for x in point)
            buckets.setdefault(key, []).append(i)

        for key, ids in buckets.items():
            if len(ids) <= 1:
                continue
            point = np.array(key, dtype=np.int32)
            if self._is_exit(point):
                allowed = ids[: self.cfg.exit_capacity]
                conflicts.update(ids[self.cfg.exit_capacity :])
                if len(allowed) == 0:
                    conflicts.update(ids)
            else:
                stayers = [i for i in ids if np.array_equal(proposals[i], self.agent_pos[i])]
                if stayers:
                    conflicts.update(i for i in ids if i not in stayers)
                else:
                    conflicts.update(ids)
        return conflicts

    def _resolve_swap_conflicts(self, prev_pos: np.ndarray, proposals: np.ndarray) -> set[int]:
        conflicts: set[int] = set()
        for i in range(self.n_agents):
            if not self.active[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if not self.active[j]:
                    continue
                i_to_j = np.array_equal(proposals[i], prev_pos[j])
                j_to_i = np.array_equal(proposals[j], prev_pos[i])
                both_moved = not np.array_equal(proposals[i], prev_pos[i]) or not np.array_equal(
                    proposals[j], prev_pos[j]
                )
                if i_to_j and j_to_i and both_moved:
                    conflicts.update([i, j])
        return conflicts

    def _resolve_occupied_target_conflicts(self, prev_pos: np.ndarray, proposals: np.ndarray) -> set[int]:
        conflicts: set[int] = set()
        for i in range(self.n_agents):
            if not self.active[i] or np.array_equal(proposals[i], prev_pos[i]):
                continue
            for j in range(self.n_agents):
                if i == j or not self.active[j]:
                    continue
                target_is_j_old_cell = np.array_equal(proposals[i], prev_pos[j])
                j_did_not_vacate = np.array_equal(proposals[j], prev_pos[j])
                if target_is_j_old_cell and j_did_not_vacate:
                    conflicts.add(i)
                    break
        return conflicts

    def _local_crowd(self, agent_id: int, positions: np.ndarray) -> int:
        if not self.active[agent_id]:
            return 0
        count = 0
        center = positions[agent_id]
        for j in range(self.n_agents):
            if j == agent_id or not self.active[j]:
                continue
            if np.max(np.abs(positions[j] - center)) <= 1:
                count += 1
        return count

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        occupied = {
            tuple(int(x) for x in self.agent_pos[i])
            for i in range(self.n_agents)
            if self.active[i]
        }
        radius = self.cfg.local_view_radius

        for i in range(self.n_agents):
            pos = self.agent_pos[i].astype(np.float32)
            delta = self._nearest_exit_delta(self.agent_pos[i])
            dist = self._distance_to_nearest_exit(self.agent_pos[i]) / max(1.0, self._diagonal)
            action_norm = (
                0.0
                if self.prev_actions[i] < 0
                else 2.0 * self.prev_actions[i] / max(1, self.n_actions - 1) - 1.0
            )
            base = [
                2.0 * pos[0] / max(1, self.rows - 1) - 1.0,
                2.0 * pos[1] / max(1, self.cols - 1) - 1.0,
                np.clip(delta[0] / max(1, self.rows), -1.0, 1.0),
                np.clip(delta[1] / max(1, self.cols), -1.0, 1.0),
                np.clip(dist, 0.0, 1.0),
                1.0 if self.active[i] else -1.0,
                action_norm,
            ]

            local_values: List[float] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    cell = np.array([self.agent_pos[i, 0] + dr, self.agent_pos[i, 1] + dc])
                    key = tuple(int(x) for x in cell)
                    if self._blocked(cell):
                        value = -1.0
                    elif self._is_exit(cell):
                        value = 1.0
                    elif key in occupied and key != tuple(self.agent_pos[i]):
                        value = 0.5
                    else:
                        value = 0.0
                    local_values.append(value)

            neighbor_values: List[float] = []
            others = []
            for j in range(self.n_agents):
                if j != i and self.active[j]:
                    d = float(np.linalg.norm(self.agent_pos[j] - self.agent_pos[i]))
                    others.append((d, j))
            others.sort(key=lambda item: item[0])
            for _, j in others[: self.cfg.max_neighbors]:
                rel = self.agent_pos[j] - self.agent_pos[i]
                neighbor_values.extend(
                    [
                        np.clip(rel[0] / max(1, self.rows), -1.0, 1.0),
                        np.clip(rel[1] / max(1, self.cols), -1.0, 1.0),
                        1.0,
                    ]
                )
            while len(neighbor_values) < 3 * self.cfg.max_neighbors:
                neighbor_values.extend([0.0, 0.0, -1.0])

            obs[i] = np.array(base + local_values + neighbor_values, dtype=np.float32)
        return obs

    def _empty_components(self) -> Dict[str, float]:
        return {
            "step": 0.0,
            "wait": 0.0,
            "distance": 0.0,
            "obstacle": 0.0,
            "collision": 0.0,
            "swap": 0.0,
            "crowd": 0.0,
            "revisit": 0.0,
            "goal": 0.0,
            "team": 0.0,
        }

    def _info(self, escaped_this_step: int) -> Dict:
        return {
            "positions": self.agent_pos.copy(),
            "active": self.active.copy(),
            "arrival_steps": self.arrival_steps.copy(),
            "escaped_this_step": int(escaped_this_step),
            "num_evacuated": int(np.sum(~self.active)),
            "paths": [[p.copy() for p in path] for path in self.paths],
        }

    def render_text(self) -> str:
        canvas = np.full((self.rows, self.cols), ".", dtype="<U2")
        canvas[self.grid_map == 1] = "#"
        for exit_point in self.exits:
            canvas[exit_point[0], exit_point[1]] = "E"
        for i, pos in enumerate(self.agent_pos):
            if self.active[i]:
                canvas[pos[0], pos[1]] = str(i % 10)
        return "\n".join("".join(row) for row in canvas)

    def render(self):
        return self.render_text()
