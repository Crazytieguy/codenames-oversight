#!/bin/bash
#SBATCH --job-name=codenames_oversight_evaluate_models
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A.out
#SBATCH --error=job_outputs/%x/%A.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH -c 1
#SBATCH --array=0-9

singularity exec --nv --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/evaluate_models.sh $SLURM_ARRAY_TASK_ID"
