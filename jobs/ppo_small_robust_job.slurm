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

#SBATCH --array=0-9

kl_coeffs=(0.06 0.07 0.08 0.09 0.105 0.12 0.14 0.16 0.185 0.215)

kl_coeff=${kl_coeffs[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/ppo_small_robust.sh 2.0e-4 $kl_coeff"
