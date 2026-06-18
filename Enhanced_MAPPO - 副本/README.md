# MAPPO Multi-Agent Evacuation

This folder is isolated from the existing project files. It implements a discrete grid multi-agent evacuation baseline with MAPPO.

## Files

- `emappo_env.py`: enhanced multi-agent evacuation environment. It keeps the same map convention as the original `Env.py`: `0` is free space and `1` is obstacle.
- `emappo.py`: enhanced shared discrete Actor plus centralized Critic MAPPO implementation.
- `train.py`: training entry point.
- `evaluate.py`: greedy rollout and path plotting.
- `visualize.py`: plotting helpers.
- `utils.py`: map loading and default start/exit selection.

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

The training and evaluation scripts now use code config by default. Edit the
settings at the top of `train.py`, `train1.py`, or `evaluate.py`, then run the
file directly from your editor.

Important settings:

- `MAP_FILE`: txt map path.
- `NUM_AGENTS`: number of agents.
- `SEED`: fixed random seed for reproducible random starts and paths.
- `EXITS = []`: read target cells marked as `2` in the txt map. If no `2` is
  present, the code falls back to border exits.
- `STARTS = []`: randomly choose `NUM_AGENTS` walkable start cells using
  `SEED`.
- `MODEL_PATH = None` in `evaluate.py`: use the newest `best_model.pt` or
  `final_model.pt` under `results`.

To manually fix exits or starts, fill them in the code:

```python
EXITS = [(0, 10), (0, 11)]
STARTS = [(20, 5), (21, 5), (22, 5)]
```

The default animation format is MP4. MP4 export requires `ffmpeg`; with conda you can install it using:

```bash
conda install ffmpeg -c conda-forge -y
```

If `ffmpeg` is not available, the code automatically falls back to GIF. To
request GIF explicitly, set `ANIMATION_FORMAT = "gif"` in the script.

## Design Notes

The environment is synchronous: all agents choose actions first, then the environment resolves wall hits, obstacle hits, same-cell conflicts, and swap conflicts. The default action set contains the original eight grid moves plus a wait action, because waiting is useful in bottlenecks and near exits.

MAPPO uses CTDE:

- Actor input: each agent's local observation, flattened from a `3 x 11 x 11` local map plus a 3D target-direction vector.
- Local map channels: obstacles, other agents, and the nearest exit/target. The other-agent channel does not include the current agent. The target channel marks the nearest exit if it is inside the local FOV; otherwise it marks a virtual target point on the FOV boundary in the nearest-exit direction.
- Target-direction vector: signed horizontal unit direction, signed vertical unit direction, and normalized Euclidean distance to the nearest exit.
- Critic input: the concatenated observations of all agents.
- Actor network: a stronger CNN map encoder plus target-vector FC fusion, without LSTM, Transformer, or blocking heads.
- Actor output: a categorical distribution over discrete grid actions.
- Critic network: a lighter per-agent encoder with mean/max context aggregation across all agents.
- Critic output: one value estimate per agent.
