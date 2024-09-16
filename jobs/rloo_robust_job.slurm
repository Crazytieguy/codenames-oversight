#!/bin/bash
#SBATCH --job-name=codenames_oversight_critiquer_only_hyperparam_sweep
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%02a.out
#SBATCH --error=job_outputs/%x/%A_%02a.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH -c 1
#SBATCH --array=0-8

learning_rates=(4.4e-5 6.7e-5 1e-4)
kl_divergences=(0.03 0.045 0.07)

learning_rate_idx=$((SLURM_ARRAY_TASK_ID % 3))
kl_divergence_idx=$((SLURM_ARRAY_TASK_ID / 3))

learning_rate=${learning_rates[$learning_rate_idx]}
kl_divergence=${kl_divergences[$kl_divergence_idx]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/rloo_robust.sh $learning_rate $kl_divergence"
