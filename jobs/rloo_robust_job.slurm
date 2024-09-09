#!/bin/bash
#SBATCH --job-name=codenames_oversight_rloo_robust_judge_hyperparam_sweep
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%02a.out
#SBATCH --error=job_outputs/%x/%A_%02a.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH -c 1
#SBATCH --array=0-6

learning_rates=(1e-4 1.5e-4 2.2e-4 3.3e-4 5e-4 7.5e-4 1.1e-3)

learning_rate=${learning_rates[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/rloo_robust.sh $learning_rate"
