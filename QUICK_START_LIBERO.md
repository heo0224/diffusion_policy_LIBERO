# Quick Start: LIBERO Task Suite Training

## One-Command Training

### Train on a Single Task Suite

```bash
# LIBERO-Spatial (10 spatial reasoning tasks)
./train_libero_spatial.sh

# LIBERO-Object (10 object manipulation tasks)
./train_libero_object.sh

# LIBERO-Goal (10 goal-conditioned tasks)
./train_libero_goal.sh

# LIBERO-90 (90 pretraining tasks)
./train_libero_90.sh

# LIBERO-10 (10 testing tasks)
./train_libero_10.sh
```

### Train on All Task Suites

```bash
# Train on all 5 task suites sequentially
./train_all_libero_suites.sh
```

## Quick Test (Limited Episodes)

```bash
# Test with only 20 episodes
./train_libero_spatial.sh task.max_episodes=20
```

## File Structure

```
diffusion_policy/
├── config/
│   ├── task/
│   │   ├── libero_spatial.yaml          # Task config for spatial suite
│   │   ├── libero_object.yaml           # Task config for object suite
│   │   ├── libero_goal.yaml             # Task config for goal suite
│   │   ├── libero_90.yaml               # Task config for 90 tasks
│   │   └── libero_10.yaml               # Task config for 10 tasks
│   ├── train_diffusion_unet_libero_spatial.yaml   # Training config
│   ├── train_diffusion_unet_libero_object.yaml
│   ├── train_diffusion_unet_libero_goal.yaml
│   ├── train_diffusion_unet_libero_90.yaml
│   └── train_diffusion_unet_libero_10.yaml
├── dataset/
│   └── libero_dataset.py                # Dataset implementation
├── train_libero_spatial.sh              # Training script for spatial
├── train_libero_object.sh               # Training script for object
├── train_libero_goal.sh                 # Training script for goal
├── train_libero_90.sh                   # Training script for 90 tasks
├── train_libero_10.sh                   # Training script for 10 tasks
└── train_all_libero_suites.sh          # Train all suites
```

## Task Suite Summary

| Suite | Tasks | Focus | Script |
|-------|-------|-------|--------|
| Spatial | 10 | Spatial relations | `train_libero_spatial.sh` |
| Object | 10 | Object types | `train_libero_object.sh` |
| Goal | 10 | Goal-conditioned | `train_libero_goal.sh` |
| 90 | 90 | Pretraining | `train_libero_90.sh` |
| 10 | 10 | Testing | `train_libero_10.sh` |

## Common Parameters

```bash
# Limit episodes for testing
./train_libero_spatial.sh task.max_episodes=50

# Change batch size
./train_libero_spatial.sh dataloader.batch_size=32

# Change learning rate
./train_libero_spatial.sh optimizer.lr=5e-5

# Change GPU
./train_libero_spatial.sh training.device=cuda:1

# Combine multiple parameters
./train_libero_spatial.sh \
    task.max_episodes=100 \
    dataloader.batch_size=32 \
    training.device=cuda:0
```

## Outputs

Training outputs go to:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_libero_{suite}_{task_name}/
```

WandB logging:
- Project: `diffusion_policy_libero`
- Run name includes timestamp and task suite

## Troubleshooting

**Out of memory?**
```bash
./train_libero_90.sh dataloader.batch_size=16
```

**Want to test quickly?**
```bash
./train_libero_spatial.sh task.max_episodes=10 training.num_epochs=100
```

**Need to resume training?**
```bash
./train_libero_spatial.sh training.resume=True
```

## Next Steps

1. **Start Training:** Pick a task suite and run the training script
2. **Monitor Progress:** Check WandB dashboard for training metrics
3. **Evaluate:** (Requires implementing LIBERO environment runner)

For detailed information, see:
- [LIBERO_TASK_SUITES.md](LIBERO_TASK_SUITES.md) - Complete guide
- [LIBERO_README.md](LIBERO_README.md) - Dataset integration details
