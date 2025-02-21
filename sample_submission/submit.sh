#!/bin/bash
#SBATCH --job-name=ClinIQLink_Eval          # Job name
#SBATCH --output=ClinIQLink_Eval.out        # Standard output and error log
#SBATCH --error=ClinIQLink_Eval.err         # Error log file
#SBATCH --time=00:20:00                     # Time limit hh:mm:ss
#SBATCH --partition=gpu                     # GPU partition (modify as per your HPC setup)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=32G                           # Memory per node
#SBATCH --ntasks=1                          # Run a single task

# Load required modules (modify as needed)
module load python/3.10  # Ensure correct Python version
module load cuda/11.8   # Load CUDA (modify based on your setup)
module load cudnn/8.6   # Load cuDNN for deep learning models

# Activate virtual environment (if using one)
source ~/venvs/cliniqlink/bin/activate  # Adjust path to your virtual environment

# Move to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run the evaluation script
python submit.py
#python submit_GPT2_example.py

# Print completion message
echo "Job Completed"
