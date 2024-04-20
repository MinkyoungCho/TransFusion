#!/bin/bash
#SBATCH --job-name=sparse        # Job name
#SBATCH --account=zmao98                 # account
#SBATCH --partition=spgpu               # Specify the A6s0 GPU partition or queue
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem-per-gpu=30G             # Memory per GPU
#SBATCH --time=20:00:00               # Maximum execution time (HH:MM:SS)
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


# Your GPU-accelerated command(s) go here
# Example: python my_gpu_script.py

# Make sure to activate your virtual environment if you have one.
eval "$(conda shell.bash hook)"
conda activate tr

# Replace the following line with the actual command(s) you want to run
# use ${SLURM_JOB_ID} to specify the id of your results, for example, 
# Example: python3 test.py > ${SLURM_JOB_ID}.out


# bash tools/dist_train.sh configs/sparsefusion_nusc_voxel_LC_r50.py 4 --resume-from work_dirs/sparsefusion_nusc_voxel_LC_r50_concat/epoch_4_concat.pth --work-dir work_dirs/sparsefusion_nusc_voxel_LC_r50_concat
# bash tools/dist_train.sh configs/sparsefusion_nusc_voxel_LC_r50.py 4 --resume-from work_dirs/sparsefusion_nusc_voxel_LC_r50/epoch_4_elsumffn.pth --work-dir work_dirs/sparsefusion_nusc_voxel_LC_r50
# bash tools/dist_train.sh configs/sparsefusion_nusc_voxel_LC_r50_elsumffn.py 4 --work-dir work_dirs/sparsefusion_nusc_voxel_LC_r50_elsumffn

python tools/test.py configs/transfusion_nusc_voxel_L.py ckpt/transfusion_L.pth --eval bbox