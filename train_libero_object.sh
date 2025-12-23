#!/bin/bash
# Train diffusion policy on LIBERO-Object task suite

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "Training on LIBERO-Object (10 tasks with different objects)"
python train.py --config-name=train_diffusion_unet_libero_object "$@"
