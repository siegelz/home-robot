# Setup Guide for Habitat OVMM Baseline Evaluation on a Compute Cluster

This guide provides instructions for setting up the environment required to run the `baseline_eval.slurm` script for evaluating the baseline agent in the Habitat OVMM challenge on a compute cluster.

## Prerequisites

- Access to a compute cluster with Slurm workload manager
- GPU nodes with CUDA support (compatible with CUDA 11.7 recommended)
- Git with LFS support (check with `git lfs version`)
- Access to Conda or Miniconda

## Setup Steps

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/siegelz/home-robot.git
cd home-robot
git checkout cos435_main
```

### 2. Set Up Conda Environment

Start by adding the following to your `~/.bashrc`:
```bash
export CONDA_ENVS_PATH=/scratch/network/$USER/.conda/envs
export CONDA_PKGS_DIRS=/scratch/network/$USER/.conda/pkgs
```

Then `source ~/.bashrc` to apply the changes.

HomeRobot requires Python 3.9. You'll need to create a conda environment with the necessary dependencies.

```bash
# Load any required modules on your cluster
module load anaconda3/2023.9
module load cudatoolkit/12.6

# For anaconda3/2023.9, set mamba as the default solver
conda config --set solver libmamba

# Create the home-robot environment
conda env create --name home-robot -f src/home_robot/environment.yml

# Activate the environment
conda activate home-robot

# Update conda environment with simulation-specific dependencies
conda env update -f src/home_robot_sim/environment.yml
```

### 3. Set Environment Variables

```bash
# Set the HOME_ROBOT_ROOT environment variable
export HOME_ROBOT_ROOT=$(pwd)
```

### 4. Run the Install Script

The install script will download submodules, model checkpoints, and build dependencies, including setting up the Habitat simulation environment:

```bash
# Make sure you're in the home-robot directory with the conda environment activated
cd $HOME_ROBOT_ROOT
./install_deps.sh
```

If you encounter issues with the install script related to CUDA compilation, you may need to:
1. Check with your cluster administrators about the correct CUDA modules to load
2. Modify the install script to work with your cluster's environment

### 5. Complete Simulation Setup

After running the install script, complete any remaining simulation setup steps:

```bash
# Install pytorch3d (if not already installed)
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install home_robot_sim library
pip install -e src/home_robot_sim
```

### 6. Download Habitat OVMM Data

```bash
# Run the OVMM install script to download all necessary data
$HOME_ROBOT_ROOT/projects/habitat_ovmm/install.sh
```

This script will:
- Download the HSSD scenes
- Download the objects and metadata
- Download the episodes
- Download and unzip the robot model

### 7. Sign in to Hugging Face

You need to sign in to Hugging Face and accept the license for using HSSD scenes:

1. Visit [https://huggingface.co/datasets/hssd/hssd-hab/tree/ovmm](https://huggingface.co/datasets/hssd/hssd-hab/tree/ovmm)
2. Sign in and accept the license
3. Use your login information when prompted during the data download process

### 8. Run the Slurm Job

```bash
# Submit the slurm job
sbatch baseline_eval.slurm
```

## Troubleshooting for Cluster Environments

1. **Module conflicts**: If you encounter module conflicts, try unloading conflicting modules before loading the required ones.

2. **Git LFS issues**: If the data download doesn't work properly, you may need to manually pull the LFS files:
   ```bash
   cd data/hssd-hab
   git lfs pull
   cd -

   cd data/objects
   git lfs pull
   cd -
   ```

3. **Storage quotas**: Check your cluster's storage quotas, as the dataset is large. You might need to use a scratch directory or request additional storage.

4. **Job resource limits**: Adjust the slurm parameters in `baseline_eval.slurm` based on your cluster's policies and the resources you have access to.

5. **Conda environment issues**: If you can't create a conda environment, check if your cluster has a module system with pre-installed PyTorch and other required packages.

## Additional Configuration Options

The `eval_baselines_agent.py` script accepts several command-line arguments that you can add to the slurm script:

- `--num_episodes`: Number of episodes to evaluate (default: all episodes in the dataset)
- `--agent_type`: Type of agent to evaluate (choices: "baseline", "random", "explore", default: "baseline")
- `--force_step`: Force to switch to a new episode after a number of steps (default: 20)
- `--data_dir`: Directory to save observation history for data collection (optional)

You can add these options to the slurm script as needed.
