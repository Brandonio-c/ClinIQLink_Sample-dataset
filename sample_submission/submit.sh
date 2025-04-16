#!/bin/bash
#SBATCH --job-name=ClinIQLink_Eval                # Job name
#SBATCH --output=ClinIQLink_Eval_%j.out             # Standard output and error log (includes job ID)
#SBATCH --error=ClinIQLink_Eval_%j.err              # Error log file (includes job ID)
#SBATCH --time=00:20:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                            # GPU partition (adjust as needed)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --cpus-per-task=8                          # Number of CPU cores per task
#SBATCH --mem=32G                                  # Memory per node
#SBATCH --ntasks=1                                 # Run a single task

# Exit immediately if a command exits with a non-zero status,
# if an undefined variable is used, or if any command in a pipeline fails.
set -euo pipefail

# Load required modules (adjust versions as needed)
module load python/3.10    # Ensure correct Python version
module load cuda/11.8      # Load CUDA for GPU-based inference
module load cudnn/8.6      # Load cuDNN for deep learning support

# Activate your virtual environment (adjust the path to your venv)
source ~/venvs/cliniqlink/bin/activate

# Change directory to where the job was submitted from
cd "$SLURM_SUBMIT_DIR"

# Determine which evaluation script to run:
#   - If no argument is provided, run both baseline and GPT evaluations sequentially.
#   - If "baseline" is provided, run the baseline evaluation (submit.py).
#   - If "gpt" is provided, run the GPT evaluation (submit_GPT2_example.py).

if [ "$#" -eq 0 ]; then
    echo "No evaluation type parameter provided. Running both baseline and GPT evaluations."
    echo "----------------------------------"
    echo "Running baseline evaluation (submit.py)..."
    python submit.py
    echo "----------------------------------"
    echo "Running GPT evaluation (submit_GPT2_example.py)..."
    python submit_GPT2_example.py
elif [ "$1" == "baseline" ]; then
    echo "Running baseline evaluation (submit.py)..."
    python submit.py
elif [ "$1" == "gpt" ]; then
    echo "Running GPT evaluation (submit_GPT2_example.py)..."
    python submit_GPT2_example.py
else
    echo "Invalid parameter: '$1'. Please use 'baseline' or 'gpt'."
    exit 1
fi

echo "Job Completed"
