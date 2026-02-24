# Script per generare un set di livelli di test indipendente dal training

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gridworld.level_generator import generate_level, save_level_to_json

# i livelli di test vengono salvati in una sottocartella separata
TEST_LEVEL_DIR = PROJECT_ROOT / "levels" / "test_set"
TEST_LEVEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_set():
    """Genera 10 livelli di test (2 per ogni difficoltà) con seed dedicati."""
    print(f"Generating 10 test levels in {TEST_LEVEL_DIR}...")

    # 2 livelli per ogni difficoltà (1-5) = 10 livelli totali
    idx = 1
    for diff in range(1, 6):
        for _ in range(2):
            seed = 10000 + idx  # seed diversi da quelli usati nel training
            lvl = generate_level(diff, seed=seed)
            lvl["name"] = f"Test_Level_{idx}_Diff_{diff}"

            fname = f"test_level_{idx}.json"
            save_level_to_json(lvl, str(TEST_LEVEL_DIR / fname))
            print(f"Generated {fname} (Difficulty: {diff})")
            idx += 1


if __name__ == "__main__":
    generate_test_set()
