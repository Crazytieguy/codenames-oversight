#!/bin/bash
#SBATCH --job-name=codenames_debate_ppo_small_robust
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%a.out
#SBATCH --error=job_outputs/%x/%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 1

#SBATCH --array=0-8

learning_rates=(1.5e-4 2.0e-4 2.5e-4)
kl_coeffs=(0.04 0.08 0.16)

learning_rate=${learning_rates[(($SLURM_ARRAY_TASK_ID / 3))]}
kl_coeff=${kl_coeffs[(($SLURM_ARRAY_TASK_ID % 3))]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/ppo_small_robust.sh $learning_rate $kl_coeff"
