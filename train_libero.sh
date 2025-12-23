#!/bin/bash
# Quick start script for training diffusion policy on LIBERO dataset

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

# Default: train on full dataset
# For testing with limited episodes, use: task.max_episodes=100
python train.py --config-name=train_diffusion_unet_libero_workspace "$@"
