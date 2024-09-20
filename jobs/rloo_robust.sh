set -e
# source /ext3/env.sh
export $(cat .env)

python -m codenames_oversight.rloo --cluer-warmup-ratio=0.001 data/rloo-6-4-50400-dataset.jsonl models/base-cluer-6-4-peft-with-vocab models/a100-rloo-6-4-test robust > data/a100-rloo-6-4-test.jsonl