set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1
kl_coeff=$2

python -m codenames_debate.ppo --learning-rate $learning_rate --kl-coeff $kl_coeff data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_small_robust/lr-$learning_rate-kl-$kl_coeff robust
