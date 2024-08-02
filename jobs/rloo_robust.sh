set -e
source /ext3/env.sh
export $(cat .env)

kl_coeff=$1
learning_rate=$2

python -m codenames_debate.ppo --kl-coeff $kl_coeff --learning-rate=$learning_rate data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/rloo_robust_hyperparam_sweep/kl-$kl_coeff-lr-$learning_rate robust
