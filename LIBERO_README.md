# LIBERO Dataset Integration for Diffusion Policy

This guide explains how to train diffusion policy models on the LIBERO dataset from HuggingFace.

## Overview

The LIBERO dataset integration adds:
1. **LIBERO Dataset Loader** ([diffusion_policy/dataset/libero_dataset.py](diffusion_policy/dataset/libero_dataset.py))
2. **Language Conditioning** via DistilBERT embeddings
3. **Configuration Files** for training

## Dataset Structure

The LIBERO dataset from HuggingFace (`physical-intelligence/libero`) contains robot manipulation episodes with:
- **Observations:**
  - `observation/state`: Low-dimensional state (8D)
  - `observation/image`: Third-person RGB camera (224x224x3)
  - `observation/wrist_image`: Wrist-mounted RGB camera (224x224x3)
- **Actions:** 7D control (position + orientation + gripper)
- **Language Prompts:** Natural language task descriptions

## Installation

### 1. Install Required Packages

```bash
# Activate your conda environment
conda activate robodiff

# Install datasets library (for HuggingFace)
pip install datasets --upgrade

# Install transformers (for language encoding)
pip install transformers
```

**Note:** If you have PyTorch < 2.1, the dataset will use a fallback hash-based language encoding instead of DistilBERT. For best results, upgrade PyTorch:

```bash
conda install pytorch>=2.1 -c pytorch
```

### 2. Verify Installation

```bash
python inspect_libero.py
```

This will download and inspect the first 2 episodes from the dataset.

## Training

### Quick Start

To train a diffusion policy on LIBERO:

```bash
python train.py --config-name=train_diffusion_unet_libero_workspace
```

### Configuration

The training configuration is located at [diffusion_policy/config/train_diffusion_unet_libero_workspace.yaml](diffusion_policy/config/train_diffusion_unet_libero_workspace.yaml).

Key parameters:
- **Task config:** `task: libero_image`
- **Horizon:** 16 timesteps
- **Observation steps:** 2
- **Action steps:** 8
- **Batch size:** 64
- **Learning rate:** 1e-4

### Task Configuration

The task configuration is in [diffusion_policy/config/task/libero_image.yaml](diffusion_policy/config/task/libero_image.yaml).

#### Shape Meta

The `shape_meta` defines the observation and action spaces:

```yaml
shape_meta:
  obs:
    image:
      shape: [3, 128, 128]  # Resized from 224x224
      type: rgb
    wrist_image:
      shape: [3, 128, 128]
      type: rgb
    agent_pos:
      shape: [8]  # Low-dimensional state
    language_embedding:
      shape: [768]  # DistilBERT embedding
  action:
    shape: [7]  # Position (3) + Orientation (3) + Gripper (1)
```

### Limiting Episodes (for Testing)

To train on a subset of data for testing:

```yaml
# In diffusion_policy/config/task/libero_image.yaml
max_episodes: 100  # Only load first 100 episodes
```

Or via command line:

```bash
python train.py --config-name=train_diffusion_unet_libero_workspace \
    task.max_episodes=100
```

## Implementation Details

### Language Conditioning

The dataset encodes language prompts using DistilBERT:
1. Each episode's prompt is tokenized
2. Encoded using DistilBERT to get a 768D embedding
3. The [CLS] token embedding is used as the language conditioning
4. Language embeddings are stored in the ReplayBuffer metadata
5. During training, language embeddings are concatenated with visual/proprioceptive observations

### Data Pipeline

1. **Load from HuggingFace:** Episodes are downloaded from `physical-intelligence/libero`
2. **Convert to ReplayBuffer:** Data is reorganized into the diffusion policy format
3. **Language Encoding:** Prompts are encoded offline during dataset initialization
4. **Sampling:** Sequences are sampled with appropriate padding for the observation and action windows

### Memory Considerations

The dataset loads all episodes into memory (zarr-based ReplayBuffer). For the full dataset:
- Consider using `max_episodes` to limit memory usage
- Use `use_cache=True` to cache the processed dataset (TODO: implement caching)

## Evaluation

Currently, the LIBERO integration focuses on training. Evaluation in the LIBERO simulation environment requires:
1. Installing LIBERO simulation environment
2. Implementing a LIBERO environment runner (similar to RobomimicImageRunner)
3. Adding rollout evaluation during training

## Files Created

- `diffusion_policy/dataset/libero_dataset.py`: Dataset implementation
- `diffusion_policy/config/task/libero_image.yaml`: Task configuration
- `diffusion_policy/config/train_diffusion_unet_libero_workspace.yaml`: Training configuration
- `inspect_libero.py`: Dataset inspection script
- `test_libero_dataset.py`: Unit tests for dataset loading

## Troubleshooting

### Issue: Transformers not loading

If you see warnings about transformers or DistilBERT not loading, the dataset will use a fallback hash-based embedding. This is deterministic but not semantically meaningful.

**Solution:** Upgrade PyTorch to >= 2.1:
```bash
conda install pytorch>=2.1 -c pytorch
```

### Issue: Out of Memory

**Solution:** Limit the number of episodes:
```bash
python train.py --config-name=train_diffusion_unet_libero_workspace \
    task.max_episodes=500
```

### Issue: Slow dataset loading

The first time you load the dataset, it needs to download from HuggingFace and process all episodes. This can take some time.

**Future improvement:** Implement caching via `use_cache=True` parameter.

## Next Steps

To fully integrate LIBERO with evaluation:

1. **Install LIBERO simulation:**
   ```bash
   pip install libero
   ```

2. **Implement LIBERO runner:**
   Create `diffusion_policy/env_runner/libero_runner.py` based on `robomimic_image_runner.py`

3. **Add evaluation config:**
   Update task config with environment runner settings

4. **Enable rollout evaluation:**
   Set `rollout_every: 50` in training config to evaluate policy periodically

## References

- [LIBERO Dataset (HuggingFace)](https://huggingface.co/datasets/physical-intelligence/libero)
- [LIBERO Paper](https://lifelong-robot-learning.github.io/LIBERO/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
