set -e
source /ext3/env.sh
export $(cat .env)

kl_coeff=$1

python -m codenames_oversight.ppo --kl-coeff $kl_coeff data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_robust_kl_sweep/kl-$kl_coeff robust
