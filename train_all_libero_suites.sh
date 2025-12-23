#!/bin/bash
# Train diffusion policy on all LIBERO task suites sequentially

source ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

echo "=========================================="
echo "Training on All LIBERO Task Suites"
echo "=========================================="
echo ""

SUITES=("spatial" "object" "goal" "90" "10")

for suite in "${SUITES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting training on LIBERO-${suite}"
    echo "=========================================="
    ./train_libero_${suite}.sh "$@"

    if [ $? -ne 0 ]; then
        echo "Error: Training on LIBERO-${suite} failed!"
        exit 1
    fi

    echo ""
    echo "Completed training on LIBERO-${suite}"
    echo ""
done

echo "=========================================="
echo "All LIBERO task suites training completed!"
echo "=========================================="
