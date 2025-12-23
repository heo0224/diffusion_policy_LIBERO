#!/usr/bin/env python
"""
Test script for LIBERO dataset integration.
"""
import sys
import torch
from omegaconf import OmegaConf

# Test basic dataset loading
def test_dataset_loading():
    print("="*50)
    print("Testing LIBERO Dataset Loading")
    print("="*50)

    # Define shape_meta
    shape_meta = {
        'obs': {
            'image': {
                'shape': [3, 128, 128],
                'type': 'rgb'
            },
            'wrist_image': {
                'shape': [3, 128, 128],
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': [8]
            },
            'language_embedding': {
                'shape': [768]
            }
        },
        'action': {
            'shape': [7]
        }
    }

    # Import dataset
    from diffusion_policy.dataset.libero_dataset import LiberoDataset

    # Create dataset with limited episodes for testing
    print("\nInitializing dataset with max_episodes=10 for testing...")
    dataset = LiberoDataset(
        shape_meta=shape_meta,
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        seed=42,
        val_ratio=0.1,
        max_train_episodes=None,
        use_cache=False,
        max_episodes=10  # Only load 10 episodes for testing
    )

    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of episodes: {dataset.replay_buffer.n_episodes}")

    # Test __getitem__
    print("\nTesting __getitem__...")
    sample = dataset[0]

    print("\nSample structure:")
    print(f"  Keys: {sample.keys()}")
    print(f"\n  Observation keys: {sample['obs'].keys()}")
    for key, value in sample['obs'].items():
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"\n  Action: shape={sample['action'].shape}, dtype={sample['action'].dtype}")

    # Test normalizer
    print("\nTesting normalizer...")
    normalizer = dataset.get_normalizer()
    print(f"Normalizer keys: {list(normalizer.keys())}")

    # Test validation split
    print("\nTesting validation split...")
    val_dataset = dataset.get_validation_dataset()
    print(f"Validation samples: {len(val_dataset)}")

    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch obs keys: {batch['obs'].keys()}")
    for key, value in batch['obs'].items():
        print(f"  {key}: shape={value.shape}")
    print(f"Batch action: shape={batch['action'].shape}")

    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
    return True

if __name__ == "__main__":
    try:
        test_dataset_loading()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
