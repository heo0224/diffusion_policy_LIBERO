#!/usr/bin/env python
"""
Quick script to inspect LIBERO dataset structure.
"""
from datasets import load_dataset
import numpy as np

print("Loading first 2 episodes from LIBERO dataset...")
ds = load_dataset('physical-intelligence/libero', split='train[:2]')

print(f"\nDataset info:")
print(f"  Number of episodes: {len(ds)}")
print(f"  Features: {ds.features}")
print(f"  Column names: {ds.column_names}")

print("\nFirst episode structure:")
episode = ds[0]
print(f"  Keys: {list(episode.keys())}")

for key, value in episode.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"  {key}: type={type(value).__name__}, length={len(value)}")
        if len(value) > 0:
            first_elem = value[0]
            if hasattr(first_elem, 'shape'):
                print(f"    - first element: shape={first_elem.shape}, dtype={first_elem.dtype}")
            else:
                print(f"    - first element: type={type(first_elem).__name__}")
    else:
        val_preview = str(value)[:100] if len(str(value)) > 100 else str(value)
        print(f"  {key}: type={type(value).__name__}, value={val_preview}")

# Check action dimension
if 'action' in episode:
    actions = episode['action']
    if hasattr(actions, 'shape'):
        print(f"\nAction details:")
        print(f"  Shape: {actions.shape}")
        print(f"  Min: {actions.min()}, Max: {actions.max()}")
        print(f"  First action: {actions[0] if len(actions) > 0 else 'N/A'}")
