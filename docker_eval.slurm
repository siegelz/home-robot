#!/bin/bash

#SBATCH --job-name=ovmm_eval
#SBATCH --output=slurm_logs/ovmm_eval-%j.out
#SBATCH --error=slurm_logs/ovmm_eval-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Set environment variables
export HOME_ROBOT_ROOT=$(pwd)
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create directory for logs
mkdir -p slurm_logs

# Navigate to the habitat_ovmm directory
cd $HOME_ROBOT_ROOT/projects/habitat_ovmm/

# Run the evaluation using Apptainer
echo "Running evaluation with baseline agent..."
apptainer run --nv \
    --bind $HOME_ROBOT_ROOT/data:/home-robot/data \
    --env "AGENT_EVALUATION_TYPE=local" \
    --env "LOCAL_ARGS=habitat.dataset.split=minival" \
    ovmm_baseline_submission.sif

echo "Evaluation complete!"
