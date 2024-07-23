set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1
init_ratio=$2

python -m codenames_debate.ppo --learning-rate $learning_rate --init-ratio $init_ratio data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_small_robust_init_ratio/lr-$learning_rate-ir-$init_ratio robust
