#!/bin/bash
#SBATCH --job-name=codenames_debate_ppo_mvp
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --error=job_outputs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 1

#SBATCH --array=0-4

learning_rates=(2e-5 4e-5 6e-5 8e-5 1e-4)

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "./ppo_mvp.sh ${learning_rates[$SLURM_ARRAY_TASK_ID]}"
