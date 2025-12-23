# LIBERO Task Suite Training Guide

This guide explains how to train separate diffusion policies for each LIBERO task suite.

## Overview

LIBERO contains 130 manipulation tasks organized into 5 task suites:

| Task Suite | Tasks | Description | Training Script |
|------------|-------|-------------|-----------------|
| **LIBERO-Spatial** | 10 | Spatial reasoning tasks with identical objects in different configurations | `train_libero_spatial.sh` |
| **LIBERO-Object** | 10 | Object manipulation tasks with different object types | `train_libero_object.sh` |
| **LIBERO-Goal** | 10 | Goal-conditioned manipulation with changing targets | `train_libero_goal.sh` |
| **LIBERO-90** | 90 | Pretraining split of LIBERO-100 (diverse manipulation tasks) | `train_libero_90.sh` |
| **LIBERO-10** | 10 | Testing split of LIBERO-100 for lifelong learning evaluation | `train_libero_10.sh` |

## Task Suite Details

### LIBERO-Spatial (libero_spatial)
- **Purpose:** Tests knowledge transfer for spatial relations
- **Characteristics:** Tasks use identical objects but require different spatial arrangements
- **Use Case:** Training policies that understand spatial reasoning and object placement

### LIBERO-Object (libero_object)
- **Purpose:** Tests knowledge transfer across different object types
- **Characteristics:** Tasks involve manipulating various objects
- **Use Case:** Training policies that generalize across object categories

### LIBERO-Goal (libero_goal)
- **Purpose:** Tests goal-conditioned manipulation
- **Characteristics:** Tasks with varying goal specifications
- **Use Case:** Training policies that can adapt to different task objectives

### LIBERO-90 & LIBERO-10
- **Purpose:** Lifelong learning benchmark
- **LIBERO-90:** Pretraining on 90 diverse manipulation tasks
- **LIBERO-10:** Held-out 10 tasks for testing lifelong learning capability
- **Use Case:** Research on continual learning and knowledge transfer

## Quick Start

### Training on a Specific Task Suite

```bash
# Train on LIBERO-Spatial
./train_libero_spatial.sh

# Train on LIBERO-Object
./train_libero_object.sh

# Train on LIBERO-Goal
./train_libero_goal.sh

# Train on LIBERO-90 (pretraining)
./train_libero_90.sh

# Train on LIBERO-10 (testing)
./train_libero_10.sh
```

### Testing with Limited Episodes

To quickly test the pipeline with limited data:

```bash
# Train on LIBERO-Spatial with only 50 episodes
./train_libero_spatial.sh task.max_episodes=50

# Train on LIBERO-90 with only 100 episodes
./train_libero_90.sh task.max_episodes=100
```

## Configuration Files

Each task suite has dedicated configuration files:

### Task Configs
Located in `diffusion_policy/config/task/`:
- `libero_spatial.yaml` - LIBERO-Spatial configuration
- `libero_object.yaml` - LIBERO-Object configuration
- `libero_goal.yaml` - LIBERO-Goal configuration
- `libero_90.yaml` - LIBERO-90 configuration
- `libero_10.yaml` - LIBERO-10 configuration

### Training Configs
Located in `diffusion_policy/config/`:
- `train_diffusion_unet_libero_spatial.yaml`
- `train_diffusion_unet_libero_object.yaml`
- `train_diffusion_unet_libero_goal.yaml`
- `train_diffusion_unet_libero_90.yaml`
- `train_diffusion_unet_libero_10.yaml`

## Advanced Usage

### Custom Training Parameters

Override any configuration parameter:

```bash
# Change batch size
./train_libero_spatial.sh dataloader.batch_size=32

# Change learning rate
./train_libero_object.sh optimizer.lr=5e-5

# Change number of epochs
./train_libero_goal.sh training.num_epochs=5000

# Multiple overrides
./train_libero_90.sh \
    dataloader.batch_size=32 \
    optimizer.lr=5e-5 \
    training.num_epochs=5000
```

### Monitoring Training

The training outputs are logged to WandB:
- Project: `diffusion_policy_libero`
- Tags include task suite name and experiment name

To change the WandB project:
```bash
./train_libero_spatial.sh logging.project=my_project_name
```

### Output Directories

Training outputs are saved to:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_libero_{suite}_{task_name}/
```

Checkpoints are saved with format:
```
epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt
```

## Typical Training Workflow

### 1. Pretraining on LIBERO-90

```bash
# Pretrain a general manipulation policy on 90 diverse tasks
./train_libero_90.sh
```

### 2. Fine-tuning on Specific Suites

After pretraining, you can fine-tune on specific task suites:

```bash
# Fine-tune on LIBERO-Spatial
./train_libero_spatial.sh training.resume=True \
    checkpoint.path=/path/to/libero_90_checkpoint.ckpt
```

### 3. Lifelong Learning Evaluation

Use LIBERO-10 for downstream lifelong learning:

```bash
# Start from LIBERO-90 pretrained checkpoint
./train_libero_10.sh training.resume=True \
    checkpoint.path=/path/to/libero_90_checkpoint.ckpt
```

## Dataset Filtering Implementation

The dataset automatically filters episodes based on the `task_suite` parameter:

```python
# In libero_dataset.py
dataset = LiberoDataset(
    shape_meta=shape_meta,
    task_suite='libero_spatial',  # Filters for spatial tasks only
    ...
)
```

The filtering works by:
1. Loading the full LIBERO dataset from HuggingFace
2. Checking each episode's task suite identifier
3. Keeping only episodes that match the requested suite
4. Creating a ReplayBuffer with filtered episodes

## Task Suite Identification

The dataset uses multiple strategies to identify task suites:
1. Check for explicit `task_suite` field in episode data
2. Infer from `task_name` or `task` field
3. Pattern matching on task descriptions

If you encounter filtering issues, the dataset will warn you and load all episodes.

## Hardware Requirements

### GPU Memory

Recommended GPU memory by task suite:
- **LIBERO-Spatial/Object/Goal:** 16GB+ (RTX 3090, A5000, etc.)
- **LIBERO-90:** 24GB+ (RTX 3090/4090, A5000, A6000, etc.)
- **LIBERO-10:** 16GB+

Reduce batch size if you encounter OOM:
```bash
./train_libero_90.sh dataloader.batch_size=32
```

### Storage

Dataset sizes (approximate):
- LIBERO-Spatial: ~5GB
- LIBERO-Object: ~5GB
- LIBERO-Goal: ~5GB
- LIBERO-90: ~45GB
- LIBERO-10: ~5GB

## Troubleshooting

### Dataset Not Filtering Correctly

If the task suite filter isn't working:
1. Check the actual field names in the HuggingFace dataset
2. Update the `_infer_task_suite` method in `libero_dataset.py`
3. Add debug prints to see episode metadata

### Out of Memory

Reduce batch size and/or image resolution:
```bash
./train_libero_90.sh \
    dataloader.batch_size=16 \
    task.shape_meta.obs.image.shape=[3,96,96] \
    task.shape_meta.obs.wrist_image.shape=[3,96,96]
```

### Slow Data Loading

The first time you load a task suite, it downloads from HuggingFace. Subsequent runs will use cached data. To speed up:
- Use `max_episodes` parameter for testing
- Increase `dataloader.num_workers` if you have CPU cores available

## Research Workflows

### Distribution Shift Analysis

Train separate policies on each controlled suite:
```bash
# Train on each suite independently
./train_libero_spatial.sh
./train_libero_object.sh
./train_libero_goal.sh

# Evaluate cross-suite generalization
# (requires implementing evaluation runner)
```

### Lifelong Learning Benchmark

Standard LIBERO benchmark protocol:
```bash
# 1. Pretrain on LIBERO-90
./train_libero_90.sh

# 2. Sequentially train on LIBERO-10 tasks
# (requires custom script for sequential task presentation)
```

## References

- **LIBERO Paper:** [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf)
- **LIBERO Website:** https://libero-project.github.io/
- **Dataset Documentation:** https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html
- **HuggingFace Dataset:** https://huggingface.co/datasets/physical-intelligence/libero

## Sources

- [LIBERO Datasets](https://libero-project.github.io/datasets)
- [LIBERO GitHub Repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO NeurIPS 2023 Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf)
- [LIBERO Documentation](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html)
- [HuggingFace LIBERO Dataset](https://huggingface.co/datasets/physical-intelligence/libero)
