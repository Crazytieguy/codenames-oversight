set -e
source /ext3/env.sh
export $(cat .env)

python -m experiments.negligent_biased_sweep < data/hard_games.jsonl > results/oracle-2048-words.jsonl
