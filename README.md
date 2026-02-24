# GridWorld — Agente RL

Un agente di Reinforcement Learning che risolve livelli GridWorld generati proceduralmente. L'agente si muove su una griglia, evita ostacoli e zone di rischio, raccoglie una chiave per aprire una porta e raggiunge l'obiettivo finale.

## Descrizione del Problema

L'ambiente è una griglia (da 7×7 a 11×11) con i seguenti elementi:

| Simbolo | Elemento | Descrizione |
|---------|----------|-------------|
| `A` | Agente | Il giocatore, parte da `(0,0)` |
| `K` | Chiave | Va raccolta prima di poter aprire la porta |
| `D` | Porta | Blocca il passaggio — serve la chiave |
| `G` | Goal | La cella obiettivo (condizione di vittoria) |
| `#` | Ostacolo | Muro invalicabile |
| `!` | Zona Rischio | Attraversabile ma con penalità |

Una barriera di muri sulla riga della porta costringe l'agente a passare obbligatoriamente per la porta, rendendo la raccolta della chiave indispensabile.

## Come Funziona

Il progetto implementa un classico problema di Reinforcement Learning in cui un agente impara, per tentativi ed errori, a navigare un ambiente a griglia con vincoli specifici.

### Il concetto di base

L'agente non conosce in anticipo la soluzione del livello. Inizialmente compie azioni casuali (esplorazione), e col tempo impara quali azioni portano a ricompense positive e quali a penalità. Questo processo è gestito dall'algoritmo **DQN (Deep Q-Network)**, che utilizza una rete neurale per stimare il valore di ogni azione possibile dato lo stato attuale.

### L'osservazione

Lo stato dell'ambiente viene codificato come un tensore a 6 canali di dimensione `(6, H, W)`, dove ogni canale è una matrice binaria che rappresenta un elemento diverso della griglia:

1. **Canale 0** — posizione dell'agente
2. **Canale 1** — ostacoli
3. **Canale 2** — chiave (scompare quando raccolta)
4. **Canale 3** — porta (valore 1.0 se chiusa, 0.5 se aperta)
5. **Canale 4** — goal
6. **Canale 5** — zone di rischio

Questa rappresentazione multi-canale permette alla CNN di "vedere" tutti gli elementi rilevanti contemporaneamente, in modo simile a come i canali RGB codificano un'immagine a colori.

### La rete neurale (GridWorldCNN)

La policy network è una CNN custom composta da:
- 3 layer convoluzionali (64 → 128 → 128 filtri, kernel 3×3) con attivazione ReLU
- Un layer lineare fully connected (512 unità) che produce le feature finali
- L'output del DQN è un vettore di Q-values, uno per ciascuna delle 4 azioni possibili (su, giù, sinistra, destra)

L'agente sceglie l'azione con il Q-value più alto (exploitation) oppure un'azione casuale (exploration), secondo una strategia ε-greedy che riduce gradualmente l'esplorazione durante il training.

### Il reward shaping

Per guidare l'apprendimento, il sistema di ricompense è strutturato in modo da incentivare comportamenti desiderabili:

| Evento | Reward |
|--------|--------|
| Ogni passo | −0.15 |
| Raccolta chiave | +3.0 |
| Raggiungimento goal (porta aperta) | +20.0 |
| Urto contro muro / ostacolo | −0.3 |
| Ingresso in zona rischio | −3.0 |
| Tentativo porta chiusa | −1.0 |
| Rivisitazione cella (3+ volte in 10 passi) | −0.2 |
| Visita cella nuova | +0.05 |

La penalità per passo incoraggia l'agente a trovare percorsi brevi, mentre il bonus per l'esplorazione lo spinge a non restare bloccato. La penalità di rivisitazione previene i loop.

## Walkthrough — Flusso di Lavoro

Ecco il percorso completo dall'installazione alla valutazione dei risultati.

### 1. Generazione dei Livelli

All'avvio del training, il sistema genera automaticamente un pack di 5 livelli di default (uno per difficoltà) nella cartella `levels/`. Ogni livello viene creato dal **generatore procedurale** (`level_generator.py`) che:

1. Sceglie la dimensione della griglia in base alla difficoltà
2. Posiziona start, chiave, porta e goal
3. Crea una barriera di muri sulla riga della porta
4. Aggiunge ostacoli casuali e zone di rischio
5. Verifica con un **BFS** (Breadth-First Search) che il livello sia risolvibile
6. Se non lo è, rigenera con un seed diverso

### 2. Training dell'Agente

Lo script `agents/train.py` gestisce l'intero processo di addestramento:

1. **Setup ambienti paralleli** — vengono creati N ambienti (default: 8) che girano in parallelo tramite `SubprocVecEnv`, accelerando la raccolta di esperienza
2. **Creazione modello** — il DQN viene inizializzato con la CNN custom e gli iperparametri configurati
3. **Ciclo di training** — l'agente interagisce con gli ambienti, raccoglie esperienze nel replay buffer e aggiorna la rete ogni 4 passi
4. **Logging** — un callback registra il reward cumulativo di ogni episodio

In modalità **mixed** (consigliata), l'80% dei livelli è generato proceduralmente (per la generalizzazione) e il 20% è preso dalla suite fissa (per la stabilità).

### 3. Valutazione

Dopo il training, lo script esegue automaticamente:

1. **Test suite** — valuta il modello sui 5 livelli di default (20 episodi ciascuno)
2. **Heatmap** — genera una mappa delle visite sul livello 3
3. **Grafico di efficacia** — un pannello 2×2 con curva di reward, success rate, confronto livelli e heatmap
4. **GIF** — replay animato dell'episodio sul livello più difficile

Per una valutazione più approfondita si usa `agents/evaluate.py`, che supporta report CSV, confronto tra modelli e test su livelli custom.

### 4. Output Generati

Al termine del training, nella cartella `output/` si trovano tutti i risultati pronti per l'analisi.

## Architettura

- **Algoritmo:** DQN (Deep Q-Network) attraverso [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Policy Network:** CNN custom (`GridWorldCNN`) — 3 layer convoluzionali (64→128→128 filtri, kernel 3×3) seguiti da un layer fully connected (512 unità)
- **Osservazione:** Griglia binaria a 6 canali di forma `(6, H, W)`
- **Strategia di Training:** Training misto (80% procedurale / 20% livelli fissi) per bilanciare generalizzazione e stabilità
- **Parallelismo:** `SubprocVecEnv` con numero configurabile di ambienti paralleli (default: 8)

### Iperparametri DQN

| Parametro | Valore |
|-----------|--------|
| Learning rate | 5×10⁻⁴ |
| Replay buffer | 500.000 |
| Learning starts | 10.000 passi |
| Exploration fraction | 0.4 |
| ε iniziale | 1.0 |
| ε finale | 0.03 |
| Batch size | 256 |
| Aggiornamento target network (τ) | 0.005 (soft update) |
| Intervallo aggiornamento target | 500 passi |
| Frequenza di training | Ogni 4 passi |
| Gradient steps | 2 |
| Fattore di sconto (γ) | 0.99 |

## Installazione

Richiede Python 3.10+.

```bash
pip install -r requirements.txt
```

## Utilizzo

### Training

```bash
# Training misto (consigliato) — 80% procedurale, 20% suite fissa
python agents/train.py --train-mixed --timesteps 5000000 --num-envs 8

# Solo livelli procedurali
python agents/train.py --train-on-procedural --timesteps 3000000

# Solo suite fissa (5 livelli)
python agents/train.py --train-on-suite --timesteps 1000000

# Riprendere il training da un checkpoint
python agents/train.py --train-mixed --timesteps 1000000 --load-model output/models/dqn_final.zip
```

### Valutazione

```bash
# Valutazione sulla test suite standard (livelli 1–5)
python agents/evaluate.py --model_path output/models/dqn_final.zip --run_suite

# Valutazione su livelli procedurali custom
python generate_test_set.py
python agents/evaluate.py --model_path output/models/dqn_final.zip --test_folder levels/test_set

# Valutazione singolo livello con salvataggio GIF
python agents/evaluate.py --model_path output/models/dqn_final.zip --level 5 --save_gif

# Valutazione deterministica (greedy)
python agents/evaluate.py --model_path output/models/dqn_final.zip --run_suite --deterministic
```

### TensorBoard

```bash
tensorboard --logdir output/logs/tensorboard
```

## Struttura del Progetto

```
├── agents/
│   ├── train.py            # Pipeline di training (DQN + SubprocVecEnv)
│   └── evaluate.py         # Valutazione, generazione GIF, confronto modelli
├── gridworld/
│   ├── __init__.py
│   ├── env.py              # GridWorldEnv (ambiente Gymnasium)
│   ├── level_generator.py  # Generazione procedurale con validazione BFS
│   └── utils.py            # I/O, plotting, funzioni di supporto
├── levels/
│   ├── level_1–5.json      # Livelli fissi per training/test
│   └── test_set/           # Livelli di test per la generalizzazione
├── tests/
│   └── test_env.py         # Test unitari dell'ambiente
├── output/
│   ├── models/             # Checkpoint del modello
│   ├── logs/               # Log di training (CSV + TensorBoard)
│   ├── plots/              # Grafici di performance
│   └── gifs/               # Replay visivi
├── generate_test_set.py    # Script per creare livelli di test
├── requirements.txt
└── README.md
```

## Generatore di Livelli

I livelli vengono generati proceduralmente su 5 livelli di difficoltà:

| Difficoltà | Griglia | Ostacoli | Zone Rischio |
|------------|---------|----------|--------------|
| 1 | 7×7 | 3 + barriera | 1 |
| 2 | 8×8 | 8 + barriera | 4 |
| 3 | 9×9 | 15 + barriera | 9 |
| 4 | 10×10 | 24 + barriera | 16 |
| 5 | 11×11 | 35 + barriera | 25 |

Ogni livello generato viene validato tramite BFS per garantire che l'agente possa raggiungere la chiave e successivamente il goal passando per la porta.

## Note

- La valutazione usa di default una policy stocastica (ε-greedy con l'ε finale del modello) per produrre varianza significativa tra gli episodi. Usare `--deterministic` per una valutazione greedy.
- La griglia di osservazione viene estesa con padding alla dimensione massima (11×11) in modo che un singolo modello possa gestire tutti i livelli di difficoltà.
- Il generatore di livelli crea una barriera di muri sull'intera riga della porta, forzando l'agente a passare per la porta.

## Risultati

Il modello è stato addestrato per **5.000.000 di timestep** in modalità mista (80% livelli procedurali, 20% suite fissa) su CUDA con 8 ambienti paralleli. Il training ha richiesto circa 77 minuti.

### Progressione del Training

| Fase (timestep) | Reward Medio | Episodi Positivi |
|------------------|--------------|------------------|
| 0 – 1.3M | −41.42 | 40.7% |
| 1.3M – 1.9M | +0.64 | 78.4% |
| 1.9M – 2.4M | +5.69 | 85.9% |
| 2.4M – 2.8M | +8.43 | 89.2% |
| 3.5M – 3.9M | +9.52 | 89.7% |
| 4.6M – 5.0M | **+9.94** | **90.3%** |
| Ultimi 1.000 episodi | **+10.72** | **90.7%** |

Episodi completati totali: **125.945**.

### Risultati Test Suite (Livelli Fissi 1–5)

| Livello | Griglia | Success Rate | Reward Medio | Passi Medi | Passi per Chiave | Chiave → Goal |
|---------|---------|:------------:|:------------:|:----------:|:----------------:|:-------------:|
| 1 | 7×7 | **100%** | 21.94 | 10.3 | 4.1 | 6.2 |
| 2 | 8×8 | **100%** | 21.31 | 14.8 | 2.2 | 12.5 |
| 3 | 9×9 | **100%** | 15.57 | 14.2 | 4.0 | 10.1 |
| 4 | 10×10 | **100%** | 15.06 | 16.8 | 9.4 | 7.3 |
| 5 | 11×11 | **100%** | 7.91 | 19.4 | 5.2 | 14.2 |
| **Totale** | | **100%** | **16.36** | **15.1** | **5.0** | **10.1** |

L'agente risolve tutti e 5 i livelli fissi con un **success rate del 100%** su 20 episodi di valutazione ciascuno, con percorsi quasi ottimali.

### Generalizzazione (Test Set Custom)

Il modello è stato valutato anche su 10 livelli procedurali mai visti (10 episodi ciascuno):

| Livello Test | Success Rate | Reward Medio | Passi Medi |
|:------------:|:------------:|:------------:|:----------:|
| 1 | **100%** | 21.96 | 10.1 |
| 2 | **100%** | 21.95 | 10.4 |
| 3 | **100%** | 21.76 | 12.4 |
| 4 | **100%** | 21.50 | 12.0 |
| 5 | **100%** | 14.90 | 14.5 |
| 6 | **100%** | 15.11 | 18.6 |
| 7 | **100%** | 7.38 | 23.4 |
| 8 | **100%** | 14.64 | 17.2 |
| 9 | **100%** | 7.34 | 21.1 |
| 10 | **100%** | 14.67 | 19.2 |
| **Totale** | **100%** | **16.12** | **15.9** |

L'agente raggiunge un **success rate del 100% su tutti i livelli mai visti**, dimostrando una forte capacità di generalizzazione su tutte le difficoltà. Come atteso, i livelli più difficili (7, 9) richiedono più passi e producono reward inferiori a causa di percorsi più lunghi e più ostacoli, ma l'agente trova sempre una soluzione.

### File di Output

| File | Descrizione |
|------|-------------|
| `output/models/dqn_final.zip` | Modello DQN addestrato |
| `output/plots/reward_curve.png` | Curva di reward durante il training |
| `output/plots/model_effectiveness.png` | Dashboard di efficacia (4 pannelli) |
| `output/gifs/best_episode.gif` | Replay visivo del miglior episodio |
| `output/trajectories/best_episode.png` | Visualizzazione della traiettoria |
| `output/logs/training_logs.csv` | Log completo di training per episodio |
| `output/test_suite_report.csv` | Risultati dettagliati della test suite |
| `output/effectiveness_report.txt` | Report riassuntivo di efficacia |
