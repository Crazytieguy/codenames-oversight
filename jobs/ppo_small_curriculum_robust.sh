set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1

python -m codenames_debate.ppo --learning-rate $learning_rate data/ppo-small-curriculum-dataset.jsonl models/llama-7b-random-cluer-small-curriculum-merged models/llama-7b-small-curriculum-ppo-robust-$learning_rate robust
