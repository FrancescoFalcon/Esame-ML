# Test per l'ambiente GridWorld, il generatore di livelli e le utility
# Si usa pytest con fixture per configurare gli ambienti di prova

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gridworld.env import GridWorldConfig, GridWorldEnv
from gridworld.level_generator import generate_level, _validate_paths, DEFAULT_LEVEL_METADATA
from gridworld import utils


@pytest.fixture
def simple_config():
    """Livello minimale 5x5 senza zone rischio."""
    return GridWorldConfig(
        grid_size=5,
        start=(0, 0),
        key=(0, 4),
        door=(2, 2),
        goal=(4, 4),
        obstacles=[(2, 0), (2, 1), (2, 3), (2, 4)],  # barriera con porta a (2,2)
        risk_zones=[],
    )


@pytest.fixture
def simple_env(simple_config):
    return GridWorldEnv(simple_config)


class TestGridWorldConfig:
    """Verifica la corretta serializzazione e i valori predefiniti della configurazione."""

    def test_from_dict_round_trip(self, simple_config):
        """Config -> dict -> Config deve dare lo stesso risultato."""
        d = simple_config.to_dict()
        restored = GridWorldConfig.from_dict(d)
        assert restored.grid_size == simple_config.grid_size
        assert restored.start == simple_config.start
        assert restored.key == simple_config.key
        assert restored.door == simple_config.door
        assert restored.goal == simple_config.goal
        assert restored.obstacles == simple_config.obstacles

    def test_defaults_consistent_with_from_dict(self):
        default_cfg = GridWorldConfig()
        minimal = {
            "grid_size": 7, "start": [0, 0], "key": [1, 1],
            "door": [3, 3], "goal": [6, 6],
        }
        from_dict_cfg = GridWorldConfig.from_dict(minimal)
        assert default_cfg.step_penalty == from_dict_cfg.step_penalty
        assert default_cfg.wall_penalty == from_dict_cfg.wall_penalty
        assert default_cfg.repetition_penalty == from_dict_cfg.repetition_penalty
        assert default_cfg.exploration_bonus == from_dict_cfg.exploration_bonus
        assert default_cfg.key_reward == from_dict_cfg.key_reward
        assert default_cfg.goal_reward == from_dict_cfg.goal_reward
        assert default_cfg.door_penalty == from_dict_cfg.door_penalty
        assert default_cfg.risk_penalty == from_dict_cfg.risk_penalty


class TestGridWorldEnv:
    """Test sulla meccanica di gioco: movimenti, collisioni, chiave-porta-goal."""

    def test_reset_returns_correct_shape(self, simple_env):
        obs, info = simple_env.reset()
        assert obs.shape == (6, 5, 5)
        assert obs.dtype == np.float32

    def test_agent_starts_at_start(self, simple_env):
        simple_env.reset()
        assert simple_env.agent_pos == (0, 0)

    def test_action_space(self, simple_env):
        assert simple_env.action_space.n == 4

    def test_move_down(self, simple_env):
        simple_env.reset()
        obs, reward, terminated, truncated, info = simple_env.step(1)
        assert simple_env.agent_pos == (1, 0)
        assert not terminated

    def test_wall_collision_stays_in_place(self, simple_env):
        simple_env.reset()
        # su da (0,0) -> fuori limiti
        obs, reward, terminated, truncated, info = simple_env.step(0)
        assert simple_env.agent_pos == (0, 0)

    def test_obstacle_collision_stays_in_place(self, simple_env):
        simple_env.reset()
        simple_env.step(1)  # giù a (1,0)
        assert simple_env.agent_pos == (1, 0)
        simple_env.step(1)  # giù verso (2,0) che è ostacolo
        assert simple_env.agent_pos == (1, 0)

    def test_key_collection(self, simple_env):
        simple_env.reset()
        assert not simple_env.has_key
        for _ in range(4):
            simple_env.step(3)  # destra
        assert simple_env.agent_pos == (0, 4)
        assert simple_env.has_key

    def test_door_without_key_blocked(self, simple_env):
        simple_env.reset()
        assert not simple_env.has_key
        simple_env.step(1)  # (1,0)
        simple_env.step(3)  # (1,1)
        simple_env.step(3)  # (1,2)
        assert simple_env.agent_pos == (1, 2)
        simple_env.step(1)  # prova ad entrare porta (2,2) senza chiave
        assert simple_env.agent_pos == (1, 2)
        assert not simple_env.door_open

    def test_door_with_key_opens(self, simple_env):
        simple_env.reset()
        for _ in range(4):
            simple_env.step(3)
        assert simple_env.has_key
        simple_env.step(1)  # (1,4)
        simple_env.step(2)  # (1,3)
        simple_env.step(2)  # (1,2)
        assert simple_env.agent_pos == (1, 2)
        simple_env.step(1)  # giù alla porta (2,2)
        assert simple_env.agent_pos == (2, 2)
        assert simple_env.door_open

    def test_full_episode_success(self, simple_env):
        # test end-to-end: chiave -> porta -> goal
        simple_env.reset()
        for _ in range(4):
            simple_env.step(3)
        assert simple_env.has_key
        # vai alla porta
        simple_env.step(1)  # (1,4)
        simple_env.step(2)  # (1,3)
        simple_env.step(2)  # (1,2)
        simple_env.step(1)  # (2,2) porta si apre
        assert simple_env.door_open
        # vai al goal (4,4)
        simple_env.step(1)  # (3,2)
        simple_env.step(1)  # (4,2)
        simple_env.step(3)  # (4,3)
        obs, reward, terminated, truncated, info = simple_env.step(3)  # (4,4) goal!
        assert terminated
        assert simple_env.agent_pos == (4, 4)

    def test_goal_without_door_open_no_win(self):
        """Se l'agente arriva al goal senza aver aperto la porta, non vince."""
        cfg = GridWorldConfig(
            grid_size=5, start=(0, 0), key=(0, 4), door=(2, 2),
            goal=(1, 1),  # goal sopra la barriera
            obstacles=[(2, 0), (2, 1), (2, 3), (2, 4)],
        )
        env = GridWorldEnv(cfg)
        env.reset()
        env.step(1)  # (1,0)
        obs, reward, terminated, truncated, info = env.step(3)  # (1,1) goal ma porta chiusa
        assert not terminated

    def test_truncation_at_max_steps(self):
        cfg = GridWorldConfig(
            grid_size=5, start=(0, 0), key=(4, 4), door=(2, 2),
            goal=(4, 0),
            obstacles=[(2, 0), (2, 1), (2, 3), (2, 4)],
            max_steps=5,
        )
        env = GridWorldEnv(cfg)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(3)
            if truncated or terminated:
                break
        assert truncated

    def test_observation_channels(self, simple_env):
        obs, _ = simple_env.reset()
        # canale 0: posizione agente
        assert obs[0][0, 0] == 1.0
        assert obs[0].sum() == 1.0
        # canale 2: chiave
        assert obs[2][0, 4] == 1.0
        # canale 3: porta
        assert obs[3][2, 2] == 1.0
        # canale 4: goal
        assert obs[4][4, 4] == 1.0

    def test_obs_grid_size_padding(self):
        cfg = GridWorldConfig(grid_size=5, start=(0, 0), key=(0, 4), door=(2, 2),
                              goal=(4, 4), obstacles=[(2, 0), (2, 1), (2, 3), (2, 4)])
        env = GridWorldEnv(cfg, obs_grid_size=8)
        obs, _ = env.reset()
        assert obs.shape == (6, 8, 8)
        assert obs[0][0, 0] == 1.0

    def test_obs_grid_size_smaller_than_grid_raises(self):
        cfg = GridWorldConfig(grid_size=5, start=(0, 0), key=(0, 4), door=(2, 2),
                              goal=(4, 4), obstacles=[])
        with pytest.raises(ValueError):
            GridWorldEnv(cfg, obs_grid_size=3)

    def test_risk_zone_penalty(self):
        cfg = GridWorldConfig(
            grid_size=5, start=(0, 0), key=(0, 4), door=(2, 2),
            goal=(4, 4), obstacles=[(2, 0), (2, 1), (2, 3), (2, 4)],
            risk_zones=[(1, 0)],
        )
        env = GridWorldEnv(cfg)
        env.reset()
        _, reward, _, _, _ = env.step(1)  # giù verso (1,0) zona rischio
        assert reward < cfg.step_penalty

    def test_exploration_bonus(self, simple_env):
        simple_env.reset()
        _, r1, _, _, _ = simple_env.step(3)  # destra a (0,1) cella nuova
        _, r2, _, _, _ = simple_env.step(2)  # sinistra torna a (0,0) già visitata
        assert r1 > r2  # prima visita deve dare più reward


class TestLevelGenerator:
    """Verifica che i livelli generati siano risolvibili e strutturalmente corretti."""

    @pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
    def test_generated_levels_are_solvable(self, difficulty):
        for seed in range(5):
            lvl = generate_level(difficulty, seed=seed * 100)
            assert _validate_paths(lvl), f"Livello diff={difficulty} seed={seed*100} non risolvibile"

    @pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
    def test_generated_level_structure(self, difficulty):
        lvl = generate_level(difficulty, seed=42)
        assert "grid_size" in lvl
        assert "start" in lvl
        assert "key" in lvl
        assert "door" in lvl
        assert "goal" in lvl
        assert "obstacles" in lvl
        assert "risk_zones" in lvl
        assert lvl["grid_size"] == DEFAULT_LEVEL_METADATA[difficulty - 1].grid_size

    def test_key_not_on_obstacle(self):
        for d in range(1, 6):
            lvl = generate_level(d, seed=99)
            obs = {tuple(o) for o in lvl["obstacles"]}
            assert tuple(lvl["key"]) not in obs

    def test_start_not_on_obstacle(self):
        for d in range(1, 6):
            lvl = generate_level(d, seed=99)
            obs = {tuple(o) for o in lvl["obstacles"]}
            assert tuple(lvl["start"]) not in obs

    def test_goal_not_on_obstacle(self):
        for d in range(1, 6):
            lvl = generate_level(d, seed=99)
            obs = {tuple(o) for o in lvl["obstacles"]}
            assert tuple(lvl["goal"]) not in obs


class TestUtils:
    """Test sulle funzioni di utilità: I/O JSON, encoder numpy, media mobile."""

    def test_save_load_level_json(self, tmp_path):
        level = {"grid_size": 5, "start": [0, 0], "key": [1, 1],
                 "door": [2, 2], "goal": [4, 4]}
        path = str(tmp_path / "test_level.json")
        utils.save_level_to_json(level, path)
        loaded = utils.load_level_from_json(path)
        assert loaded == level

    def test_numpy_encoder(self):
        data = {"val": np.int64(42), "arr": np.array([1, 2, 3])}
        result = json.dumps(data, cls=utils.NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == 42
        assert parsed["arr"] == [1, 2, 3]

    def test_moving_average(self):
        vals = list(range(100))
        ma = utils.moving_average(vals, window=10)
        assert len(ma) == 91
        assert ma[0] == pytest.approx(4.5)

    def test_moving_average_short(self):
        vals = [1.0, 2.0, 3.0]
        ma = utils.moving_average(vals, window=50)
        assert len(ma) == 3

    def test_aggregate_success_rates(self):
        results = {"level_1": [1.0, 1.0, 0.0], "level_2": [0.0, 0.0]}
        agg = utils.aggregate_success_rates(results)
        assert agg["level_1"] == pytest.approx(2 / 3)
        assert agg["level_2"] == pytest.approx(0.0)

    def test_aggregate_success_rates_empty(self):
        results = {"level_1": []}
        agg = utils.aggregate_success_rates(results)
        assert agg["level_1"] == 0.0
