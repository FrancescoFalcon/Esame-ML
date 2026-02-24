# Funzioni di utilità varie (I/O, plot, ecc.)
# Raccolte qui per evitare duplicazioni tra i moduli del progetto

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Serve per serializzare tipi numpy in JSON."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path):
    """Crea la directory (e i genitori) se non esiste già."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_level_to_json(level, path):
    """Serializza un livello in formato JSON, gestendo i tipi numpy."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(level, f, indent=2, cls=NumpyEncoder)


def load_level_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_grid_heatmap(grid, title, output_path, cmap="coolwarm", show_colorbar=False):
    # genera un'immagine heatmap della griglia e la salva su disco
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reward_curve(timesteps, rewards, output_path, window=100):
    """Grafico della curva di reward con media mobile sovrapposta."""
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, rewards, alpha=0.15, color="tab:blue", label="Raw")
    if len(rewards) >= window:
        smoothed = moving_average(rewards, window=window)
        plt.plot(timesteps[window - 1:], smoothed, color="tab:blue",
                 linewidth=2, label=f"Media mobile ({window} ep.)")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_report(path, lines):
    # scrive un report testuale riga per riga
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def aggregate_success_rates(results):
    """Calcola il tasso di successo medio per ogni livello."""
    return {lvl: float(np.mean(vals)) if vals else 0.0
            for lvl, vals in results.items()}


def moving_average(values, window=50):
    # se la lista è più corta della finestra, la restituisce così com'è
    if len(values) < window:
        return np.array(values, dtype=float)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window
