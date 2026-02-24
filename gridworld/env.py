# Ambiente GridWorld custom con meccanica chiave-porta-goal
# Implementazione basata su Gymnasium, con osservazioni multi-canale

import json
import os
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import utils


# Parametri di configurazione del livello: dimensioni, posizioni degli
# elementi e valori di reward/penalità usati durante il training
@dataclass
class GridWorldConfig:
    grid_size: int = 7
    start: tuple = (0, 0)
    key: tuple = (0, 0)
    door: tuple = (0, 0)
    goal: tuple = (0, 0)
    obstacles: list = field(default_factory=list)
    risk_zones: list = field(default_factory=list)
    max_steps: int = None
    door_penalty: float = -1.0
    risk_penalty: float = -3.0
    step_penalty: float = -0.15
    wall_penalty: float = -0.3
    repetition_penalty: float = -0.2
    exploration_bonus: float = 0.05
    key_reward: float = 3.0
    goal_reward: float = 20.0

    @classmethod
    def from_dict(cls, data):
        return cls(
            grid_size=data["grid_size"],
            start=tuple(data["start"]),
            key=tuple(data["key"]),
            door=tuple(data["door"]),
            goal=tuple(data["goal"]),
            obstacles=[tuple(c) for c in data.get("obstacles", [])],
            risk_zones=[tuple(c) for c in data.get("risk_zones", [])],
            max_steps=data.get("max_steps"),
            door_penalty=data.get("door_penalty", -1.0),
            risk_penalty=data.get("risk_penalty", -3.0),
            step_penalty=data.get("step_penalty", -0.15),
            wall_penalty=data.get("wall_penalty", -0.3),
            repetition_penalty=data.get("repetition_penalty", -0.2),
            exploration_bonus=data.get("exploration_bonus", 0.05),
            key_reward=data.get("key_reward", 3.0),
            goal_reward=data.get("goal_reward", 20.0),
        )

    def to_dict(self):
        return {
            "grid_size": self.grid_size,
            "start": list(self.start),
            "key": list(self.key),
            "door": list(self.door),
            "goal": list(self.goal),
            "obstacles": [list(c) for c in self.obstacles],
            "risk_zones": [list(c) for c in self.risk_zones],
            "max_steps": self.max_steps,
            "door_penalty": self.door_penalty,
            "risk_penalty": self.risk_penalty,
            "step_penalty": self.step_penalty,
            "wall_penalty": self.wall_penalty,
            "repetition_penalty": self.repetition_penalty,
            "exploration_bonus": self.exploration_bonus,
            "key_reward": self.key_reward,
            "goal_reward": self.goal_reward,
        }


class GridWorldEnv(gym.Env):
    """Ambiente GridWorld con porta chiusa che richiede una chiave.

    L'agente deve raccogliere la chiave, aprire la porta e raggiungere
    il goal evitando ostacoli e zone di rischio.
    """

    metadata = {"render.modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, config=None, level_path=None, seed=None, obs_grid_size=None):
        # si può passare direttamente un config oppure il path al JSON del livello
        super().__init__()
        if config is None:
            if level_path is None:
                raise ValueError("Provide either config or level_path")
            config = GridWorldConfig.from_dict(utils.load_level_from_json(level_path))
        self.config = config
        self.grid_size = config.grid_size
        self.obs_grid_size = obs_grid_size or config.grid_size
        if self.obs_grid_size < self.grid_size:
            raise ValueError("obs_grid_size must be >= config.grid_size")

        # azioni: 0=su, 1=giù, 2=sx, 3=dx
        self.action_space = spaces.Discrete(4)
        # 6 canali: agente, ostacoli, chiave, porta, goal, zone rischio
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(6, self.obs_grid_size, self.obs_grid_size),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(seed)
        self.agent_pos = config.start
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.max_steps = config.max_steps or (self.grid_size * self.grid_size * 2)
        self.visitation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.recent_positions = []       # finestra scorrevole per rilevare loop
        self.visited_cells = set()

    def reset(self, *, seed=None, options=None):
        # nel caso venga passata una nuova config tramite options,
        # aggiorniamo la griglia al volo (utile per il training procedurale)
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "config" in options:
            self.config = GridWorldConfig.from_dict(options["config"])
            self.grid_size = self.config.grid_size
            if self.grid_size > self.obs_grid_size:
                raise ValueError("Config grid_size exceeds observation grid size")
            self.max_steps = self.config.max_steps or (self.grid_size * self.grid_size * 2)

        self.agent_pos = self.config.start
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.visitation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.recent_positions = [self.agent_pos]
        self.visited_cells = {self.agent_pos}

        obs = self._get_obs()
        return obs, self._build_info()

    def step(self, action):
        # logica principale: gestione collisioni, raccolta chiave, apertura porta
        assert self.action_space.contains(action)
        reward = self.config.step_penalty  # penalità base ad ogni passo
        terminated = False
        truncated = False

        next_pos = self._next_position(action)
        moved = False

        if not self._in_bounds(next_pos):
            reward += self.config.wall_penalty
        elif next_pos in self.config.obstacles:
            reward += self.config.wall_penalty
        elif next_pos == self.config.door:
            if self.has_key:
                self.door_open = True
                self.agent_pos = next_pos
                moved = True
            else:
                reward += self.config.door_penalty
        else:
            self.agent_pos = next_pos
            moved = True

        if moved:
            # penalità loop: se la stessa cella appare 3+ volte nelle ultime 10 posizioni
            self.recent_positions.append(self.agent_pos)
            if len(self.recent_positions) > 10:
                self.recent_positions.pop(0)

            if self.recent_positions.count(self.agent_pos) >= 3:
                reward += self.config.repetition_penalty

            # bonus esplorazione per celle nuove
            if self.agent_pos not in self.visited_cells:
                reward += self.config.exploration_bonus
                self.visited_cells.add(self.agent_pos)

            self.visitation[self.agent_pos] += 1

            if self.agent_pos == self.config.key and not self.has_key:
                self.has_key = True
                reward += self.config.key_reward
            if self.agent_pos in self.config.risk_zones:
                reward += self.config.risk_penalty
            if self.agent_pos == self.config.goal and self.door_open:
                reward += self.config.goal_reward
                terminated = True

        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._build_info()

    def _next_position(self, action):
        r, c = self.agent_pos
        if action == 0:    # su
            r -= 1
        elif action == 1:  # giù
            r += 1
        elif action == 2:  # sinistra
            c -= 1
        elif action == 3:  # destra
            c += 1
        return r, c

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _get_obs(self):
        # costruisce il tensore di osservazione 6xHxW con padding se necessario
        obs = np.zeros((6, self.obs_grid_size, self.obs_grid_size), dtype=np.float32)
        obs[0][self.agent_pos] = 1.0
        for cell in self.config.obstacles:
            obs[1][cell] = 1.0
        if not self.has_key:
            obs[2][self.config.key] = 1.0
        obs[3][self.config.door] = 1.0 if not self.door_open else 0.5
        obs[4][self.config.goal] = 1.0
        for cell in self.config.risk_zones:
            obs[5][cell] = 1.0
        return obs

    def _build_info(self):
        return {
            "agent_pos": self.agent_pos,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }

    def render(self):
        grid = self._symbolic_grid()
        out = "\n".join(" ".join(row) for row in grid)
        print(out)
        return out

    def _symbolic_grid(self):
        # griglia testuale per il render: A=agente, K=chiave, D=porta, G=goal, #=muro, !=rischio
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r, c in self.config.obstacles:
            grid[r][c] = "#"
        for r, c in self.config.risk_zones:
            grid[r][c] = "!"
        if not self.has_key:
            kr, kc = self.config.key
            grid[kr][kc] = "K"
        dr, dc = self.config.door
        grid[dr][dc] = "D" if not self.door_open else "d"
        gr, gc = self.config.goal
        grid[gr][gc] = "G"
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"
        return grid

    def save_trajectory_png(self, trajectory, output_path):
        # salva la traiettoria come heatmap: utile per analizzare il comportamento post-training
        grid = np.zeros((self.grid_size, self.grid_size))
        for r, c in self.config.obstacles:
            grid[r, c] = -1
        for r, c in self.config.risk_zones:
            grid[r, c] = -0.5
        for idx, (r, c) in enumerate(trajectory):
            grid[r, c] = idx + 1
        utils.render_grid_heatmap(
            grid, title="Traiettoria agente",
            output_path=output_path, cmap="viridis", show_colorbar=True,
        )

    @classmethod
    def from_json(cls, path):
        data = utils.load_level_from_json(path)
        return cls(GridWorldConfig.from_dict(data))

    def to_json(self, path):
        utils.save_level_to_json(self.config.to_dict(), path)


def register_env(env_id="GridWorld-Themed-v0"):
    # registra l'ambiente nel registry di Gymnasium (evita duplicati)
    if env_id in gym.registry:
        return
    gym.register(id=env_id, entry_point="gridworld.env:GridWorldEnv")
