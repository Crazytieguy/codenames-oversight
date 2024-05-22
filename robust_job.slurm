#!/bin/bash
#SBATCH --job-name=codenames_debate_robust
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --error=job_outputs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 1

#SBATCH --array=0-4

games_per_phase=$((4096 + $SLURM_ARRAY_TASK_ID * 1024))
singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "./robust_run.sh $games_per_phase"
