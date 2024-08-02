#!/bin/bash
#SBATCH --job-name=codenames_debate_rloo_robust_hyperparam_sweep
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%a.out
#SBATCH --error=job_outputs/%x/%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 1

#SBATCH --array=0-8

kl_coeffs=(0.033 0.057 0.1)
learning_rates=(2.5e-5 5e-5 1e-4)

task_id=$SLURM_ARRAY_TASK_ID
kl_coeff=${kl_coeffs[$task_id % 3]}
learning_rate=${learning_rates[$task_id / 3]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/rloo_robust.sh $kl_coeff $learning_rate"
