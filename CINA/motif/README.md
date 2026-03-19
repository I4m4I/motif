# Motif-Regularized Reinforcement Learning

This folder preserves the motif-regularized reinforcement-learning code used in the paper workflow. The code trains a recurrent PPO agent with an optional motif-regularization pretraining stage.

Files:

- `main.py`: training entry point
- `tools.py`: motif regularizer, recurrent actor/critic, PPO utilities
- `run_batch.sh`: multi-GPU batch launcher
- `params.ini`: example environment configuration

Run:

```bash
cd CINA/motif
python main.py --env ip --seed 1 --prefix Vanilla --cuda 0 --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
```
