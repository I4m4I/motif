# Motif-Regularized Reinforcement Learning (Fig. 3)

The motif-regularized reinforcement-learning module trains a recurrent PPO agent with an
optional motif-regularization pretraining stage. Its trained runs feed the ANN vs SNN
benchmark in [`reinforcement_learning.md`](reinforcement_learning.md) (Fig. 4).

## Components

| Path | Role |
|---|---|
| [`training/train_rl.py`](../training/train_rl.py) | Training entry point. |
| [`models/rl/motif_rl.py`](../models/rl/motif_rl.py) | Motif regularizer, recurrent actor/critic, PPO utilities. |
| [`scripts/run_rl_training.sh`](../scripts/run_rl_training.sh) | Multi-GPU batch launcher. |
| [`configs/rl_params.ini`](../configs/rl_params.ini) | Example environment configuration. |

## Run

From the repository root:

```bash
python training/train_rl.py --env ip --seed 1 --prefix Vanilla --cuda 0 \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

Or launch the full multi-GPU batch:

```bash
bash scripts/run_rl_training.sh
```

`train_rl.py` reads its environment configuration from
[`configs/rl_params.ini`](../configs/rl_params.ini) and imports the RL utilities from
[`models/rl/motif_rl.py`](../models/rl/motif_rl.py).
