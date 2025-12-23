#!/bin/bash
# Train diffusion policy on LIBERO-10 task suite (testing split of LIBERO-100)

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "Training on LIBERO-10 (10 test tasks from LIBERO-100)"
python train.py --config-name=train_diffusion_unet_libero_10 "$@"
