#!/bin/bash
#SBATCH --job-name=codenames_debate_ppo_robust_kl_sweep
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%a.out
#SBATCH --error=job_outputs/%x/%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 1
#SBATCH --array=0-7

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/ppo_negligent_biased.sh $SLURM_ARRAY_TASK_ID"
