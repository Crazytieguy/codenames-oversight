set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1

python -m codenames_debate.ppo --learning-rate $learning_rate data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_small_robust_calibrated_2/lr-$learning_rate robust
