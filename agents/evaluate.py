# Valutazione modelli RL per GridWorld
# Supporta DQN e PPO, con suite di test, GIF e confronto tra modelli

import argparse
import json
import os
from pathlib import Path
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gridworld.env import GridWorldConfig, GridWorldEnv
from gridworld.level_generator import DEFAULT_LEVEL_METADATA, build_default_level_pack
from gridworld import utils

LEVEL_DIR = PROJECT_ROOT / "levels"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_TXT = OUTPUT_DIR / "effectiveness_report.txt"
REPORT_CSV = OUTPUT_DIR / "test_suite_report.csv"
COMPARE_CSV = OUTPUT_DIR / "model_comparison.csv"
COMPARE_PNG = OUTPUT_DIR / "model_comparison.png"
GIF_DIR = OUTPUT_DIR / "gifs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MAX_OBS_GRID = max(m.grid_size for m in DEFAULT_LEVEL_METADATA)


def ensure_levels():
    """Assicura che i livelli di default esistano nella directory."""
    build_default_level_pack(str(LEVEL_DIR))


def detect_algo(model_path, requested=None):
    """Determina l'algoritmo del modello: prima dal parametro esplicito,
    poi dal file .meta.json, infine dal nome del file."""
    if requested:
        return requested.lower()
    meta_path = Path(f"{model_path}.meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("algo", "dqn")
    name = Path(model_path).name.lower()
    if "ppo" in name:
        return "ppo"
    return "dqn"


def load_model(model_path, algo=None):
    # carica il modello usando la classe corretta in base all'algoritmo
    alg = detect_algo(model_path, algo)
    if alg == "ppo":
        model = PPO.load(model_path)
    elif alg == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Algoritmo non supportato: {alg}")
    return model, alg


def make_env(level_path):
    cfg = utils.load_level_from_json(level_path)
    gc = GridWorldConfig.from_dict(cfg)
    obs_grid = max(gc.grid_size, MAX_OBS_GRID)
    return GridWorldEnv(gc, obs_grid_size=obs_grid)


def play_episode(model, env, deterministic=True, capture_frames=False, seed=None):
    """Esegue un singolo episodio e raccoglie metriche dettagliate."""
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    tot_reward = 0.0
    steps = 0
    steps_to_key = None
    steps_to_goal = None
    door_step = None
    fail_reason = "max_steps"
    frames = []
    trajectory = [env.agent_pos]
    action_counts = {}  # distribuzione delle azioni scelte dall'agente

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        a = int(action)
        action_counts[a] = action_counts.get(a, 0) + 1

        obs, reward, done, truncated, info = env.step(a)
        tot_reward += reward
        steps += 1
        if info.get("has_key") and steps_to_key is None:
            steps_to_key = steps
        if info.get("door_open") and door_step is None:
            door_step = steps
        if done:
            steps_to_goal = steps
            fail_reason = "success"
        elif truncated:
            fail_reason = "timeout"
        if capture_frames:
            frames.append(_grid_to_image(env._symbolic_grid(), step=steps))
        trajectory.append(env.agent_pos)

    return {
        "success": done,
        "reward": tot_reward,
        "steps": steps,
        "steps_to_key": steps_to_key,
        "steps_key_to_goal": (steps_to_goal - steps_to_key
                              if steps_to_goal and steps_to_key else None),
        "door_open_step": door_step,
        "failure_reason": fail_reason,
        "trajectory": trajectory,
        "frames": frames,
    }


def evaluate_model(model, env, episodes=5, deterministic=True, base_seed=0):
    """Valuta il modello su N episodi e restituisce statistiche aggregate."""
    stats = []
    for i in range(episodes):
        res = play_episode(model, env, deterministic=deterministic,
                           capture_frames=False, seed=base_seed + i)
        stats.append(res)
    sr = np.mean([1 if r["success"] else 0 for r in stats])
    mean_r = np.mean([r["reward"] for r in stats])
    std_r = np.std([r["reward"] for r in stats])
    return {
        "success_rate": float(sr),
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "episodes": stats,
    }


def evaluate_and_print_trajectory(model, env, episodes=5, deterministic=True):
    for ep in range(episodes):
        print(f"\nEpisodio {ep + 1}")
        obs, _ = env.reset()
        done, truncated = False, False
        steps = 0
        while not (done or truncated):
            env.render()
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1
        env.render()
        print(f"Terminato in {steps} step | Successo: {done}")


def save_replay_gif(model, env, output_path, deterministic=True):
    # cattura tutti i frame di un episodio e li salva come GIF animata
    result = play_episode(model, env, deterministic=deterministic, capture_frames=True)
    frames = result["frames"]
    if not frames:
        raise RuntimeError("Nessun frame catturato per la GIF")
    utils.ensure_dir(Path(output_path).parent)
    imageio.mimsave(output_path, frames, duration=0.4)
    return output_path


def run_custom_suite(model_path, level_files, algo=None,
                     num_episodes=10, deterministic=False):
    model, alg = load_model(model_path, algo)
    records = []

    print(f"Valutazione su {len(level_files)} livelli custom...")
    for lpath in level_files:
        try:
            env = make_env(str(lpath))
            name = lpath.stem
            for ep in range(num_episodes):
                res = play_episode(model, env, deterministic=deterministic,
                                   capture_frames=False, seed=ep)
                records.append({
                    "level": name,
                    "episode": ep,
                    "success": res["success"],
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "steps_to_key": res["steps_to_key"],
                    "steps_key_to_goal": res["steps_key_to_goal"],
                    "failure_reason": res["failure_reason"],
                })
        except Exception as e:
            print(f"Errore nel valutare {lpath}: {e}")

    df = pd.DataFrame(records)
    utils.ensure_dir(OUTPUT_DIR)
    csv_path = OUTPUT_DIR / "custom_suite_report.csv"
    df.to_csv(csv_path, index=False)

    sr = df["success"].mean()
    print(f"Successo complessivo suite custom: {sr:.2f}")

    return {"records": df, "success_rate": sr, "report_csv": str(csv_path)}


def run_test_suite(model_path, algo=None, num_episodes=20, deterministic=False):
    """Esegue la valutazione completa sui 5 livelli di default."""
    ensure_levels()
    model, alg = load_model(model_path, algo)
    records = []

    for meta in DEFAULT_LEVEL_METADATA:
        lpath = LEVEL_DIR / f"level_{meta.difficulty}.json"
        env = make_env(str(lpath))
        for ep in range(num_episodes):
            res = play_episode(model, env, deterministic=deterministic,
                               capture_frames=False, seed=ep)
            records.append({
                "level": meta.difficulty,
                "episode": ep,
                "success": res["success"],
                "reward": res["reward"],
                "steps": res["steps"],
                "steps_to_key": res["steps_to_key"],
                "steps_key_to_goal": res["steps_key_to_goal"],
                "failure_reason": res["failure_reason"],
            })

    df = pd.DataFrame(records)
    utils.ensure_dir(OUTPUT_DIR)
    df.to_csv(REPORT_CSV, index=False)
    summary = _summarize_suite(df)
    utils.write_report(REPORT_TXT, summary)
    success_rates = df.groupby("level")["success"].mean().to_dict()
    return {
        "records": df,
        "success_rates": success_rates,
        "report_csv": str(REPORT_CSV),
        "report_txt": str(REPORT_TXT),
        "algo": alg,
    }


def compare_models(model_paths, algo=None):
    """Confronta più modelli sulla stessa suite e produce un grafico riassuntivo."""
    results = []
    for p in model_paths:
        suite = run_test_suite(p, algo=algo)
        rates = suite["success_rates"]
        results.append({"model": Path(p).name,
                        **{f"level_{k}": v for k, v in rates.items()}})
    df = pd.DataFrame(results)
    df.to_csv(COMPARE_CSV, index=False)
    utils.ensure_dir(PLOTS_DIR)

    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        lvls = sorted(int(c.split("_")[1]) for c in row.index if c.startswith("level_"))
        vals = [row[f"level_{l}"] for l in lvls]
        plt.plot(lvls, vals, marker="o", label=row["model"])
    plt.xticks(lvls)
    plt.ylim(0, 1)
    plt.xlabel("Livello")
    plt.ylabel("Success rate")
    plt.title("Comparazione modelli")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARE_PNG)
    plt.close()
    return str(COMPARE_CSV)


def _grid_to_image(grid, step):
    # converte la griglia simbolica in un'immagine RGBA per la GIF
    fig, ax = plt.subplots(figsize=(3, 3))
    table = ax.table(cellText=grid, loc="center", cellLoc="center")
    table.scale(1, 2)
    ax.axis("off")
    ax.set_title(f"Step {step}")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img


def _summarize_suite(df):
    """Produce un riepilogo testuale dei risultati della suite."""
    lines = ["=== Test Suite GridWorld ==="]
    fail_counts = df["failure_reason"].value_counts().to_dict()
    for lvl, grp in df.groupby("level"):
        sr = grp["success"].mean()
        avg_steps = grp["steps"].mean()
        avg_rew = grp["reward"].mean()
        kt = grp["steps_to_key"].dropna().mean()
        bridge = grp["steps_key_to_goal"].dropna().mean()
        lines.append(
            f"Livello {lvl}: success={sr:.2f}, steps={avg_steps:.1f}, "
            f"reward={avg_rew:.2f}, key={kt:.1f}, key->goal={bridge:.1f}"
        )
    tot = df["success"].mean()
    lines.append(f"Successo complessivo: {tot:.2f}")
    lines.append(f"Reward medio globale: {df['reward'].mean():.2f} +/- {df['reward'].std():.2f}")
    lines.append(f"Passi medi per la chiave: {df['steps_to_key'].dropna().mean():.1f}")
    lines.append(f"Passi medi chiave->porta->goal: {df['steps_key_to_goal'].dropna().mean():.1f}")
    lines.append("Note fallimenti: " + ", ".join(f"{k}={v}" for k, v in fail_counts.items()))
    return lines


def parse_args():
    p = argparse.ArgumentParser(description="Valutazione GridWorld")
    p.add_argument("--model_path", type=str, nargs="?",
                   help="Percorso del modello SB3")
    p.add_argument("--algo", type=str, choices=["dqn", "ppo"], nargs="?",
                   help="Algoritmo del modello")
    p.add_argument("--level", type=int, choices=range(1, 6),
                   help="Livello singolo da valutare")
    p.add_argument("--level_file", type=str,
                   help="Percorso file JSON livello specifico")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic policy (greedy). Default is stochastic.")
    p.add_argument("--run_suite", action="store_true")
    p.add_argument("--test_folder", type=str,
                   help="Cartella con livelli JSON per test custom")
    p.add_argument("--compare_models", nargs="*",
                   help="Confronta modelli molteplici")
    p.add_argument("--save_gif", action="store_true")
    return p.parse_args()


def main():
    # gestisce le varie modalità: singolo livello, suite, confronto, GIF
    args = parse_args()
    ensure_levels()

    if args.compare_models:
        compare_models(args.compare_models, algo=args.algo)
        return

    if not args.model_path:
        raise SystemExit("Serve --model_path o --compare_models")

    model, algo = load_model(args.model_path, args.algo)

    if args.run_suite:
        run_test_suite(args.model_path, algo=algo)

    if args.test_folder:
        folder = Path(args.test_folder)
        if folder.exists() and folder.is_dir():
            levels = sorted(list(folder.glob("*.json")))
            run_custom_suite(args.model_path, levels, algo=algo,
                             num_episodes=args.episodes,
                             deterministic=args.deterministic)
        else:
            print(f"Cartella non trovata: {args.test_folder}")
        return

    if args.level_file:
        lpath = Path(args.level_file)
    elif args.level:
        lpath = LEVEL_DIR / f"level_{args.level}.json"
    else:
        lpath = LEVEL_DIR / "level_1.json"

    env = make_env(str(lpath))
    stats = evaluate_model(model, env, episodes=args.episodes,
                           deterministic=args.deterministic)
    print(json.dumps(stats, indent=2))

    if args.save_gif:
        gif_path = GIF_DIR / "best_episode.gif"
        save_replay_gif(model, env, str(gif_path))


if __name__ == "__main__":
    main()
