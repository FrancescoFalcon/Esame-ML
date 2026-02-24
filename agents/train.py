# Script di training per il progetto GridWorld
# Utilizza DQN con CNN custom e supporto multi-environment

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gridworld import utils
from gridworld.env import GridWorldConfig, GridWorldEnv
from gridworld.level_generator import (
    DEFAULT_LEVEL_METADATA, build_default_level_pack, generate_level,
)

# directory di output per modelli, log, grafici e traiettorie
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
GIF_DIR = OUTPUT_DIR / "gifs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
LEVEL_DIR = PROJECT_ROOT / "levels"

# la griglia di osservazione deve coprire il livello più grande disponibile
MAX_OBS_GRID = max(m.grid_size for m in DEFAULT_LEVEL_METADATA)


class GridWorldCNN(BaseFeaturesExtractor):
    """CNN custom per le osservazioni GridWorld (6, H, W).

    Tre blocchi convoluzionali seguiti da un layer lineare.
    La dimensione del flatten viene calcolata automaticamente.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # calcolo automatico della dimensione dopo flatten
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


class RewardLoggerCallback(BaseCallback):
    """Callback che registra il reward cumulativo di ogni episodio terminato."""

    def __init__(self):
        super().__init__()
        self.record = []
        self._ep_rewards = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n, dtype=np.float64)

    def _on_step(self):
        rews = self.locals["rewards"]
        dones = self.locals["dones"]
        self._ep_rewards += rews
        for i, done in enumerate(dones):
            if done:
                self.record.append({"timesteps": self.num_timesteps,
                                    "reward": float(self._ep_rewards[i])})
                self._ep_rewards[i] = 0.0
        return True


class ProceduralWrapper(gym.Wrapper):
    """Wrapper che genera un livello procedurale ad ogni reset.

    Ogni volta che l'ambiente viene resettato, si crea un livello
    nuovo con difficoltà casuale nell'intervallo indicato.
    """
    def __init__(self, env, min_diff, max_diff, seed=None):
        super().__init__(env)
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        diff = int(self.rng.integers(self.min_diff, self.max_diff + 1))
        cfg = generate_level(diff, seed=int(self.rng.integers(0, 1_000_000)))
        return self.env.reset(options={"config": cfg}, **kwargs)


class RandomFixedLevelWrapper(gym.Wrapper):
    """Wrapper che sceglie un livello fisso random ad ogni reset."""
    def __init__(self, env, level_dir, levels, seed=None):
        super().__init__(env)
        self.level_dir = level_dir
        self.levels = levels
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        idx = self.rng.choice(self.levels)
        path = self.level_dir / f"level_{idx}.json"
        cfg = utils.load_level_from_json(str(path))
        return self.env.reset(options={"config": cfg}, **kwargs)


class MixedWrapper(gym.Wrapper):
    """Mix di livelli procedurali (80%) e fissi (20%)."""
    def __init__(self, env, level_dir, min_diff, max_diff, seed=None, procedural_prob=0.8):
        super().__init__(env)
        self.level_dir = level_dir
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.rng = np.random.default_rng(seed)
        self.fixed_levels = [1, 2, 3, 4, 5]
        self.procedural_prob = procedural_prob

    def reset(self, **kwargs):
        if self.rng.random() < self.procedural_prob:
            diff = int(self.rng.integers(self.min_diff, self.max_diff + 1))
            cfg = generate_level(diff, seed=int(self.rng.integers(0, 1_000_000)))
        else:
            idx = self.rng.choice(self.fixed_levels)
            path = self.level_dir / f"level_{idx}.json"
            cfg = utils.load_level_from_json(str(path))
        return self.env.reset(options={"config": cfg}, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description="Training GridWorld")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-size", type=int, default=None)
    p.add_argument("--difficulty", type=int, choices=range(1, 6), default=1)
    p.add_argument("--train-on-procedural", action="store_true")
    p.add_argument("--train-on-suite", action="store_true",
                   help="Train on the fixed suite of levels 1-5")
    p.add_argument("--train-mixed", action="store_true",
                   help="Train on both procedural and fixed suite (80/20)")
    p.add_argument("--num-envs", type=int, default=8,
                   help="Number of parallel environments")
    p.add_argument("--config", type=str, help="Path to custom level JSON")
    p.add_argument("--load-model", type=str,
                   help="Path to pretrained model to resume training")
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


def ensure_assets():
    """Crea le directory necessarie e genera il level pack di default."""
    for d in [MODELS_DIR, LOGS_DIR, GIF_DIR, PLOTS_DIR, TRAJ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    build_default_level_pack(str(LEVEL_DIR))


def load_level_config(args):
    if args.config:
        return utils.load_level_from_json(args.config)
    if args.train_on_procedural:
        return generate_level(args.difficulty, seed=args.seed)
    path = LEVEL_DIR / f"level_{args.difficulty}.json"
    if not path.exists():
        return generate_level(1, seed=42)
    return utils.load_level_from_json(str(path))


def make_env(args, base_config):
    # dimensione griglia di osservazione: deve essere almeno pari alla più grande
    obs_grid = max(base_config.get("grid_size", MAX_OBS_GRID), MAX_OBS_GRID)
    n_envs = args.num_envs

    def _make_init(rank):
        def _init():
            env = GridWorldEnv(GridWorldConfig.from_dict(base_config),
                               obs_grid_size=obs_grid)
            if args.train_mixed:
                env = MixedWrapper(env, LEVEL_DIR, 1, 5,
                                   seed=args.seed + rank, procedural_prob=0.8)
            elif args.train_on_procedural:
                env = ProceduralWrapper(env, 1, 5, seed=args.seed + rank)
            elif args.train_on_suite:
                env = RandomFixedLevelWrapper(env, LEVEL_DIR, [1, 2, 3, 4, 5],
                                             seed=args.seed + rank)
            return env
        return _init

    fns = [_make_init(i) for i in range(n_envs)]
    # con più ambienti si usa SubprocVecEnv per il parallelismo reale
    if n_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def create_model(env, tensorboard, seed, load_path=None,
                 is_procedural=False, is_mixed=False):
    """Crea o carica il modello DQN con la CNN customizzata."""
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[Train] Using {device.upper()} for policy updates")

    if load_path:
        print(f"[Train] Loading model from {load_path}")
        model = DQN.load(load_path, env=env, device=device)
        if is_procedural or is_mixed:
            model.exploration_initial_eps = 0.15
            model.exploration_final_eps = 0.02
            model.exploration_fraction = 0.2
            print("[Train] Adjusted exploration for generalization recovery")
        else:
            model.exploration_initial_eps = 0.2
            model.exploration_final_eps = 0.02
            model.exploration_fraction = 0.2
            print("[Train] Adjusted exploration for fine-tuning")
        model.exploration_schedule = get_linear_fn(
            model.exploration_initial_eps,
            model.exploration_final_eps,
            model.exploration_fraction,
        )
        return model

    # iperparametri della policy: si usa la nostra CNN come feature extractor
    policy_kwargs = dict(
        features_extractor_class=GridWorldCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    tb_log = str(LOGS_DIR / "tensorboard") if tensorboard else None

    model = DQN(
        "CnnPolicy", env,
        learning_rate=5e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.03,
        batch_size=256,
        tau=0.005,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=2,
        gamma=0.99,
        tensorboard_log=tb_log,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        device=device,
    )
    return model


def save_metadata(model_path, algo):
    """Salva un file .meta.json accanto al modello per ricordare l'algoritmo usato."""
    meta_path = Path(f"{model_path}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"algo": algo}, f, indent=2)


def export_training_logs(records):
    df = pd.DataFrame(records)
    csv_path = LOGS_DIR / "training_logs.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_reward(records):
    if not records:
        raise ValueError("Nessun record di reward")
    ts = [r["timesteps"] for r in records]
    rews = [r["reward"] for r in records]
    out = PLOTS_DIR / "reward_curve.png"
    utils.plot_reward_curve(ts, rews, str(out))
    return out


def capture_heatmap(model):
    """Esegue un episodio sul livello 3 e restituisce la heatmap delle visite."""
    level3 = LEVEL_DIR / "level_3.json"
    env = GridWorldEnv(
        GridWorldConfig.from_dict(utils.load_level_from_json(str(level3))),
        obs_grid_size=MAX_OBS_GRID,
    )
    obs, _ = env.reset()
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(int(action))
    vis = env.visitation.astype(float)
    if vis.max() > 0:
        vis /= vis.max()
    return vis


def build_effectiveness_plot(reward_history, suite_success, heatmap):
    """Genera un grafico 2x2 con reward, success rate, confronto e heatmap."""
    out = PLOTS_DIR / "model_effectiveness.png"
    ts = [r["timesteps"] for r in reward_history]
    rews = [r["reward"] for r in reward_history]
    levels = sorted(suite_success.keys())
    sr = [suite_success.get(l, 0.0) for l in levels]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(ts, rews, color="tab:blue")
    axes[0, 0].set_title("Reward per episodio")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(levels, sr, marker="o", color="tab:green")
    axes[0, 1].set_title("Success rate suite")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Livello")
    axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(levels, sr, color="tab:orange")
    axes[1, 0].set_title("Confronto livello")
    axes[1, 0].set_xlabel("Livello")
    axes[1, 0].set_ylabel("Success rate")
    axes[1, 0].set_ylim(0, 1)

    im = axes[1, 1].imshow(heatmap, cmap="magma")
    axes[1, 1].set_title("Heatmap visite livello 3")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.label_outer()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def auto_generate_gif(model, algo):
    """Genera automaticamente una GIF dell'episodio sul livello più difficile."""
    from agents import evaluate as eval_mod

    hardest = LEVEL_DIR / "level_5.json"
    env = eval_mod.make_env(str(hardest))
    gif_path = GIF_DIR / "best_episode.gif"
    try:
        eval_mod.save_replay_gif(model, env, str(gif_path))
        traj = eval_mod.play_episode(
            model, env, deterministic=True, capture_frames=False
        )["trajectory"]
        env.save_trajectory_png(traj, str(TRAJ_DIR / "best_episode.png"))
        return str(gif_path)
    except Exception:
        return None


def main():
    # flusso principale: parsing argomenti, setup ambiente, training, valutazione
    args = parse_args()
    ensure_assets()

    base_config = load_level_config(args)
    env = make_env(args, base_config)
    callback = RewardLoggerCallback()
    model = create_model(env, args.tensorboard, args.seed, args.load_model,
                         is_procedural=args.train_on_procedural,
                         is_mixed=args.train_mixed)

    model.learn(total_timesteps=args.timesteps, callback=callback)

    model_path = MODELS_DIR / "dqn_final.zip"
    model.save(str(model_path))
    save_metadata(model_path, "dqn")

    export_training_logs(callback.record)
    plot_reward(callback.record)

    # valutazione automatica sulla suite di livelli standard
    from agents import evaluate as eval_mod
    suite = eval_mod.run_test_suite(str(model_path), algo="dqn")

    heatmap = capture_heatmap(model)
    build_effectiveness_plot(
        callback.record,
        {int(k): v for k, v in suite["success_rates"].items()},
        heatmap,
    )
    auto_generate_gif(model, "dqn")

    # riepilogo finale
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total timesteps:    {args.timesteps}")
    print(f"Parallel envs:      {args.num_envs}")
    print(f"Episodes completed: {len(callback.record)}")
    if callback.record:
        all_rews = [r["reward"] for r in callback.record]
        last = all_rews[-100:] if len(all_rews) >= 100 else all_rews
        print(f"Mean reward (last 100 ep): {np.mean(last):.2f} ± {np.std(last):.2f}")
    print(f"\nModel saved to:     {model_path}")
    print(f"Training logs:      {LOGS_DIR / 'training_logs.csv'}")
    print(f"Reward curve:       {PLOTS_DIR / 'reward_curve.png'}")
    print(f"Effectiveness plot: {PLOTS_DIR / 'model_effectiveness.png'}")
    print("\nTest Suite Results:")
    for lvl, rate in sorted(suite["success_rates"].items(), key=lambda x: int(x[0])):
        print(f"  Level {lvl}: {rate:.0%}")
    overall = np.mean(list(suite["success_rates"].values()))
    print(f"  Overall:  {overall:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
