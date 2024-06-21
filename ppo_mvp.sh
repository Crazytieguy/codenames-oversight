set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1

python -m codenames_debate.ppo data/ppo-mvp-dataset.jsonl models/llama-7b-cluer-with-targets-ppo models/llama-7b-cluer-with-targets-ppo-$learning_rate --learning-rate $learning_rate
