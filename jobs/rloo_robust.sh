set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1

>&2 echo "learning_rate: $learning_rate"

python -m codenames_oversight.rloo --critiquer-learning-rate=$learning_rate --critique-model-dir models/base-critiquer-peft data/rloo-small-22400-dataset.jsonl models/sft/negligent-biased/cluer/nw-0-bnw-0-bnnw-0-bf-1.00-aa-0.0 models/rloo-small-robust-judge-lr-2/$learning_rate robust > data/rloo-small-robust-judge-lr-2/$learning_rate.jsonl
