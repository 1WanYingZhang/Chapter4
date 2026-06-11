# MAPPO Multi-Agent Evacuation

This folder is isolated from the existing project files. It implements a discrete grid multi-agent evacuation baseline with MAPPO.

## Files

- `mappo_env.py`: multi-agent evacuation environment. It keeps the same map convention as the original `Env.py`: `0` is free space and `1` is obstacle.
- `mappo.py`: shared discrete Actor plus centralized Critic MAPPO implementation.
- `train.py`: training entry point.
- `evaluate.py`: greedy rollout and path plotting.
- `visualize.py`: plotting helpers.
- `utils.py`: map loading and default start/exit selection.

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run with the built-in demo map:

```bash
python train.py --num-agents 8 --total-timesteps 50000
```

Run with your own txt map:

```bash
python train.py --map-file "../2-code_LGB-QL_PP/your_map.txt" --num-agents 10 --exits "0,10;0,11"
```

Optional manual starts:

```bash
python train.py --map-file "../2-code_LGB-QL_PP/your_map.txt" --starts "20,5;21,5;22,5" --exits "0,10"
```

Evaluate:

```bash
python evaluate.py --model "results/<run_id>/best_model.pt" --num-agents 8
```

Save a playable evacuation video:

```bash
python evaluate.py --model "results/<run_id>/best_model.pt" --num-agents 8 --save-animation
```

The default animation format is MP4. MP4 export requires `ffmpeg`; with conda you can install it using:

```bash
conda install ffmpeg -c conda-forge -y
```

If `ffmpeg` is not available, the code automatically falls back to GIF. You can also request GIF explicitly:

```bash
python train.py --num-agents 8 --animation-format gif
```

## Design Notes

The environment is synchronous: all agents choose actions first, then the environment resolves wall hits, obstacle hits, same-cell conflicts, and swap conflicts. The default action set contains the original eight grid moves plus a wait action, because waiting is useful in bottlenecks and near exits.

MAPPO uses CTDE:

- Actor input: each agent's local observation.
- Critic input: the concatenated observations of all agents.
- Actor output: a categorical distribution over discrete grid actions.
