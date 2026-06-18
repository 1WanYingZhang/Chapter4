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
    local_view_radius: int = 5
    max_neighbors: int = 4
    allow_wait: bool = True
    exit_capacity: int = 1
    step_penalty: float = -0.05
    wait_penalty: float = -0.03
    obstacle_penalty: float = -1.0
    collision_penalty: float = -2.0
    goal_reward: float = 5.0
    non_target_exit_reward_ratio: float = 0.3
    team_completion_reward: float = 10.0


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
        self.exit_keys = {tuple(int(v) for v in exit_point) for exit_point in self.exits}

        self.local_cells = (2 * self.cfg.local_view_radius + 1) ** 2
        self.map_channels = 3
        self.goal_vector_dim = 3
        self.obs_dim = self.map_channels * self.local_cells + self.goal_vector_dim
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
        self.assigned_exits = np.zeros((self.n_agents, 2), dtype=np.int32)
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
        self.assigned_exits = np.asarray(
            [self._nearest_exit(pos) for pos in self.agent_pos],
            dtype=np.int32,
        )
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

            if self._move_blocked(prev_pos[i], move):
                proposals[i] = prev_pos[i]
                rewards[i] += self.cfg.obstacle_penalty
                components[i]["obstacle"] += self.cfg.obstacle_penalty
            else:
                proposals[i] = target

        proposals, failed_agents, _ = self._resolve_motion_conflicts(prev_pos, proposals)
        for i in failed_agents:
            if not self.active[i]:
                continue
            penalty = self.cfg.collision_penalty
            rewards[i] += penalty
            components[i]["collision"] += penalty

        escaped_this_step = 0
        for i in range(self.n_agents):
            if not self.active[i]:
                self.paths[i].append(self.agent_pos[i].copy())
                continue

            new_key = tuple(int(x) for x in proposals[i])
            self.visited[i].add(new_key)

            self.agent_pos[i] = proposals[i]
            if self._is_exit(self.agent_pos[i]):
                self.active[i] = False
                self.arrival_steps[i] = self.steps
                if self._is_assigned_exit(i, self.agent_pos[i]):
                    exit_reward = self.cfg.goal_reward
                    components[i]["target_exit"] += exit_reward
                else:
                    exit_reward = self.cfg.goal_reward * self.cfg.non_target_exit_reward_ratio
                    components[i]["non_target_exit"] += exit_reward
                rewards[i] += exit_reward
                escaped_this_step += 1

            self.paths[i].append(self.agent_pos[i].copy())

        if escaped_this_step > 0 and np.all(~self.active):
            team_bonus = self.cfg.team_completion_reward / max(1, self.n_agents)
            rewards += team_bonus
            for item in components:
                item["team"] += team_bonus

        terminated = bool(np.all(~self.active))
        truncated = bool(self.steps >= self.cfg.max_steps)
        info = self._info(escaped_this_step=escaped_this_step, include_paths=terminated)
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

    def _move_blocked(self, start: np.ndarray, move: np.ndarray) -> bool:
        target = start + move
        if self._blocked(target):
            return True

        dr, dc = int(move[0]), int(move[1])
        if dr != 0 and dc != 0:
            row_side = start + np.array([dr, 0], dtype=np.int32)
            col_side = start + np.array([0, dc], dtype=np.int32)
            if self._blocked(row_side) and self._blocked(col_side):
                return True

        return False

    def available_actions(self) -> np.ndarray:
        masks = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for i in range(self.n_agents):
            if not self.active[i]:
                if self.cfg.allow_wait:
                    masks[i, self.n_actions - 1] = 1.0
                else:
                    masks[i, :] = 1.0
                continue

            for action, move in enumerate(self.moves):
                if not self._move_blocked(self.agent_pos[i], move):
                    masks[i, action] = 1.0

            if not np.any(masks[i]):
                fallback_action = self.n_actions - 1 if self.cfg.allow_wait else 0
                masks[i, fallback_action] = 1.0
        return masks

    def _is_exit(self, point: np.ndarray) -> bool:
        return tuple(int(x) for x in point) in self.exit_keys

    def _is_assigned_exit(self, agent_id: int, point: np.ndarray) -> bool:
        return bool(np.array_equal(point, self._target_exit_for_agent(agent_id)))

    def _nearest_exit(self, point: np.ndarray) -> np.ndarray:
        deltas = self.exits - point
        idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
        return self.exits[idx]

    def _distance_to_nearest_exit(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self._nearest_exit(point) - point))

    def _nearest_exit_delta(self, point: np.ndarray) -> np.ndarray:
        return (self._nearest_exit(point) - point).astype(np.float32)

    def _target_exit_for_agent(self, agent_id: int) -> np.ndarray:
        return self.assigned_exits[agent_id]

    def _distance_to_assigned_exit(self, agent_id: int, point: np.ndarray) -> float:
        return float(np.linalg.norm(self._target_exit_for_agent(agent_id) - point))

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
            if key in self.exit_keys:
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
        prev_lookup = {
            tuple(int(x) for x in prev_pos[i]): i
            for i in range(self.n_agents)
            if self.active[i]
        }
        for i in range(self.n_agents):
            if not self.active[i]:
                continue
            j = prev_lookup.get(tuple(int(x) for x in proposals[i]))
            if j is None or j <= i or not self.active[j]:
                continue
            j_to_i = np.array_equal(proposals[j], prev_pos[i])
            both_moved = not np.array_equal(proposals[i], prev_pos[i]) or not np.array_equal(
                proposals[j], prev_pos[j]
            )
            if j_to_i and both_moved:
                conflicts.update([i, j])
        return conflicts

    def _resolve_occupied_target_conflicts(self, prev_pos: np.ndarray, proposals: np.ndarray) -> set[int]:
        conflicts: set[int] = set()
        occupied_by_stayer = {
            tuple(int(x) for x in prev_pos[j]): j
            for j in range(self.n_agents)
            if self.active[j] and np.array_equal(proposals[j], prev_pos[j])
        }
        for i in range(self.n_agents):
            if not self.active[i] or np.array_equal(proposals[i], prev_pos[i]):
                continue
            target_owner = occupied_by_stayer.get(tuple(int(x) for x in proposals[i]))
            if target_owner is not None and target_owner != i:
                conflicts.add(i)
        return conflicts

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        occupied = {
            tuple(int(x) for x in self.agent_pos[i]): i
            for i in range(self.n_agents)
            if self.active[i]
        }
        radius = self.cfg.local_view_radius

        for i in range(self.n_agents):
            target_exit = self._target_exit_for_agent(i)
            delta = (target_exit - self.agent_pos[i]).astype(np.float32)
            dy = float(delta[0])
            dx = float(delta[1])
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                dx_unit = dx / dist
                dy_unit = dy / dist
            else:
                dx_unit = 0.0
                dy_unit = 0.0
            goal_vector = np.array(
                [
                    np.clip(dx_unit, -1.0, 1.0),
                    np.clip(dy_unit, -1.0, 1.0),
                    np.clip(dist / max(1.0, self._diagonal), 0.0, 1.0),
                ],
                dtype=np.float32,
            )

            local_maps = np.zeros(
                (self.map_channels, 2 * radius + 1, 2 * radius + 1),
                dtype=np.float32,
            )
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    cell = np.array([self.agent_pos[i, 0] + dr, self.agent_pos[i, 1] + dc])
                    key = tuple(int(x) for x in cell)
                    local_r = dr + radius
                    local_c = dc + radius
                    if self._blocked(cell):
                        local_maps[0, local_r, local_c] = 1.0
                    if not self._out_of_bounds(cell):
                        if key in occupied and occupied[key] != i:
                            local_maps[1, local_r, local_c] = 1.0

            target_delta = target_exit - self.agent_pos[i]
            target_dr = int(target_delta[0])
            target_dc = int(target_delta[1])
            if abs(target_dr) <= radius and abs(target_dc) <= radius:
                target_local_r = radius + target_dr
                target_local_c = radius + target_dc
            else:
                scale = radius / max(abs(target_dr), abs(target_dc), 1)
                target_local_r = radius + int(round(target_dr * scale))
                target_local_c = radius + int(round(target_dc * scale))
                target_local_r = int(np.clip(target_local_r, 0, 2 * radius))
                target_local_c = int(np.clip(target_local_c, 0, 2 * radius))
            local_maps[2, target_local_r, target_local_c] = 1.0

            obs[i] = np.concatenate([local_maps.reshape(-1), goal_vector])
        return obs

    def _empty_components(self) -> Dict[str, float]:
        return {
            "step": 0.0,
            "wait": 0.0,
            "obstacle": 0.0,
            "collision": 0.0,
            "target_exit": 0.0,
            "non_target_exit": 0.0,
            "team": 0.0,
        }

    def _info(self, escaped_this_step: int, include_paths: bool = False) -> Dict:
        info = {
            "positions": self.agent_pos.copy(),
            "active": self.active.copy(),
            "arrival_steps": self.arrival_steps.copy(),
            "assigned_exits": self.assigned_exits.copy(),
            "available_actions": self.available_actions(),
            "escaped_this_step": int(escaped_this_step),
            "num_evacuated": int(np.sum(~self.active)),
        }
        if include_paths:
            info["paths"] = [[p.copy() for p in path] for path in self.paths]
        return info

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
