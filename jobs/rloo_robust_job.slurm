#!/bin/bash
#SBATCH --job-name=codenames_oversight_rloo_robust_hyperparam_sweep
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%02a.out
#SBATCH --error=job_outputs/%x/%A_%02a.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH -c 1
#SBATCH --array=0-24

kl_coeffs=(0.028 0.036 0.047 0.062 0.08)
learning_rates=(1.1e-5 1.4e-5 1.9e-5 2.5e-5 3.3e-5)

task_id=$SLURM_ARRAY_TASK_ID
kl_coeff=${kl_coeffs[$((task_id / 5))]}
learning_rate=${learning_rates[$((task_id % 5))]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/rloo_robust.sh $kl_coeff $learning_rate"
