#!/bin/bash
#SBATCH --job-name=codenames_debate_neglect_last_n
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x/%A_%a.out
#SBATCH --error=job_outputs/%x/%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH -c 1

singularity exec --overlay /scratch/yt3114/conda.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "python -m experiments.negligent_biased_sweep < data/hard-games.jsonl > results/oracle-2048-words.jsonl"
