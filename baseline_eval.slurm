#!/bin/bash

#SBATCH --job-name=baseline_eval
#SBATCH --output=slurm_logs/baseline_eval-%j.out
#SBATCH --error=slurm_logs/baseline_eval-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load necessary modules
module load anaconda3/2023.9
module load cudatoolkit/11.7

# Set environment variables
export HOME_ROBOT_ROOT=$(pwd)
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create directory for logs
mkdir -p slurm_logs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate home-robot

# Navigate to the habitat_ovmm directory
cd $HOME_ROBOT_ROOT/projects/habitat_ovmm/

# Run the evaluation
python eval_baselines_agent.py \
    --evaluation_type local \
    --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent.yaml \
    --env_config_path projects/habitat_ovmm/configs/env/hssd_eval.yaml

echo "Evaluation complete!"
