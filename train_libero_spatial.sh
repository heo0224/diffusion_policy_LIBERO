#!/bin/bash
# Train diffusion policy on LIBERO-Spatial task suite

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "Training on LIBERO-Spatial (10 tasks with spatial reasoning)"
python train.py --config-name=train_diffusion_unet_libero_spatial "$@"
