#!/bin/bash
# Train diffusion policy on LIBERO-90 task suite (pretraining split of LIBERO-100)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "Training on LIBERO-90 (90 pretraining tasks from LIBERO-100)"
python train.py --config-name=train_diffusion_unet_libero_90 "$@"
