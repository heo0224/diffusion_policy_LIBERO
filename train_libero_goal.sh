#!/bin/bash
# Train diffusion policy on LIBERO-Goal task suite

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "Training on LIBERO-Goal (10 tasks with goal-conditioned manipulation)"
python train.py --config-name=train_diffusion_unet_libero_goal "$@"
