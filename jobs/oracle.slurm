#!/bin/bash
#SBATCH --job-name=codenames_oversight_oracle
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A.out
#SBATCH --error=job_outputs/%x/%A.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH -c 1

singularity exec --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "jobs/oracle.sh"
