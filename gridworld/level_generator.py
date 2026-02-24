# Generazione procedurale dei livelli per GridWorld
# Ogni livello ha una difficoltà da 1 a 5 con griglia progressivamente più grande

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import utils


# struttura che descrive i parametri base di ciascun livello di difficoltà
@dataclass
class LevelMeta:
    difficulty: int
    grid_size: int
    base_obstacles: int
    base_risks: int
    description: str


# metadati dei 5 livelli di default, usati sia per la generazione
# procedurale che per la suite di valutazione
DEFAULT_LEVEL_METADATA = [
    LevelMeta(1, 7, 3, 1, "Introduzione"),
    LevelMeta(2, 8, 4, 2, "Corridoi stretti"),
    LevelMeta(3, 9, 5, 3, "Labirinto moderato"),
    LevelMeta(4, 10, 6, 4, "Labirinto avanzato"),
    LevelMeta(5, 11, 7, 5, "Sfida finale"),
]


def build_default_level_pack(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for meta in DEFAULT_LEVEL_METADATA:
        lvl = generate_level(meta.difficulty, seed=meta.difficulty)
        path = out / f"level_{meta.difficulty}.json"
        save_level_to_json(lvl, str(path))


def save_level_to_json(level, path):
    utils.save_level_to_json(level, path)


def load_level_from_json(path):
    return utils.load_level_from_json(path)


def generate_level(difficulty, seed=None):
    """Genera un livello con la difficoltà specificata.

    Posiziona start, chiave, porta e goal, poi aggiunge ostacoli
    e zone di rischio. Se il risultato non è risolvibile, riprova.
    """
    meta = DEFAULT_LEVEL_METADATA[difficulty - 1]
    rng = np.random.default_rng(seed)
    gs = meta.grid_size

    start = (0, 0)
    door_row = gs // 2
    door_col = rng.integers(gs // 3, gs - gs // 3)
    door = (door_row, int(door_col))
    goal = (gs - 2, gs - 2)
    key_row = rng.integers(1, door_row - 1)
    key_col = rng.integers(1, gs - 2)
    key = (int(key_row), int(key_col))

    # piazzamento ostacoli e zone rischio garantendo la raggiungibilità
    obstacles = _generate_obstacles(meta, rng, gs, door, start, goal, key)
    risk_zones = _generate_risks(meta, rng, gs, {door, start, goal, key}, obstacles)

    level = {
        "name": f"Procedural_Level_{difficulty}",
        "difficulty": difficulty,
        "grid_size": gs,
        "start": list(start),
        "key": list(key),
        "door": list(door),
        "goal": list(goal),
        "obstacles": [list(c) for c in obstacles],
        "risk_zones": [list(c) for c in risk_zones],
        "max_steps": gs * gs * 2,
    }

    # se il livello non è risolvibile, riprova con un seed diverso
    if not _validate_paths(level):
        return generate_level(difficulty, seed=(seed or 0) + 1337)
    return level


def _generate_obstacles(meta, rng, grid_size, door, start, goal, key):
    # il numero di ostacoli scala col quadrato della difficoltà
    num_obstacles = meta.base_obstacles * meta.difficulty
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    forbidden = {start, goal, door, key}
    door_row = door[0]

    # barriera solida sulla riga della porta, così la porta è l'unico passaggio
    barrier = {
        (door_row, c)
        for c in range(grid_size)
        if (door_row, c) not in forbidden and c != door[1]
    }

    obstacles = set(barrier)
    target = len(barrier) + num_obstacles
    while len(obstacles) < target:
        cell = tuple(rng.choice(cells))
        if cell in forbidden or cell == door:
            continue
        if cell in obstacles:
            continue
        if cell[0] == door_row:
            continue
        obstacles.add(cell)
    return sorted(obstacles)


def _generate_risks(meta, rng, grid_size, critical, obstacles):
    # le zone rischio vengono piazzate in celle libere, evitando posizioni critiche
    n = meta.base_risks * meta.difficulty
    risks = set()
    occupied = set(obstacles) | critical
    while len(risks) < n:
        cell = (rng.integers(0, grid_size), rng.integers(0, grid_size))
        if cell in occupied:
            continue
        risks.add(cell)
    return sorted(risks)


def _validate_paths(level):
    """BFS per verificare che start->key e key->goal siano raggiungibili."""
    gs = level["grid_size"]
    start = tuple(level["start"])
    key = tuple(level["key"])
    door = tuple(level["door"])
    goal = tuple(level["goal"])
    obstacles = {tuple(c) for c in level["obstacles"]}

    def bfs(src, dst, door_closed):
        queue = [src]
        visited = {src}
        while queue:
            r, c = queue.pop(0)
            if (r, c) == dst:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < gs and 0 <= nc < gs):
                    continue
                nxt = (nr, nc)
                if nxt in visited or nxt in obstacles:
                    continue
                if door_closed and nxt == door:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return False

    return bfs(start, key, True) and bfs(key, goal, False)
