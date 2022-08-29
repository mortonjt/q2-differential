#!/bin/sh
#SBATCH --job-name=q2
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2080@nyu.edu
#SBATCH --output=slurm_%j.out

python r2_score.py
