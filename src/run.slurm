#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
#SBATCH -A m4259
#SBATCH -o job_output_%j.txt
#SBATCH -e job_error_%j.txt

srun -n 4 ./ExtreM_distributed
