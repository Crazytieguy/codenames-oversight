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

#SBATCH --array=0-7

learning_rates=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "./ppo_mvp.sh ${learning_rates[$SLURM_ARRAY_TASK_ID]}"
