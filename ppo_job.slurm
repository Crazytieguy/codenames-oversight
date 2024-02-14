#!/bin/bash
#SBATCH --job-name=codenames_debate_ppo
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --error=job_outputs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/yt3114/llm-math-debate.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
export $(cat .env)
python -m codenames_debate.ppo
"
