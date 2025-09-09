#!/bin/bash
#SBATCH --job-name=gc_ttt
#SBATCH --gpus 1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=3:00:00

# Activate venv inside the job
source ~/gc_ttt/venv/bin/activate
cd ~/gc_ttt

# Run
srun python main.py default.yaml
