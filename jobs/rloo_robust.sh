set -e
source /ext3/env.sh
export $(cat .env)

kl_coeff=$1
learning_rate=$2

>&2 echo "kl_coeff: $kl_coeff"
>&2 echo "learning_rate: $learning_rate"

python -m codenames_debate.rloo --kl-coeff=$kl_coeff --learning-rate=$learning_rate data/ppo-small-32768-dataset.jsonl models/llama-7b-random-cluer-small-merged models/rloo_robust_hyperparam_sweep-32768/kl-$kl_coeff-lr-$learning_rate robust
