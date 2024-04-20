#!/bin/bash
#SBATCH --job-name=tran        # Job name
#SBATCH --account=zmao98                 # account
#SBATCH --partition=spgpu               # Specify the A6s0 GPU partition or queue
#SBATCH --gres=gpu:4                  # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem-per-gpu=40G             # Memory per GPU
#SBATCH --time=47:50:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=output/test_job.%j.out    # Output file
#SBATCH --error=output/test_job.%j.err     # Error file
#SBATCH --mail-user minkycho@umich.edu
#SBATCH --mail-type ALL



# Load necessary modules (if required)
# Example: Loading CUDA module
module load cuda/11.3.0
module load cudnn/11.3-v8.2.1
module load gcc/8.2.0
module load openmpi

# Make sure to activate your virtual environment if you have one.
eval "$(conda shell.bash hook)"
conda activate tr


tools/dist_train.sh configs/transfusion_nusc_voxel_LC.py 4
