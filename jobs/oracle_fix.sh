set -e
source /ext3/env.sh
export $(cat .env)

python -m experiments.negligent_biased_sweep_fix < data/hard_games.jsonl > results/oracle-2048-fix.jsonl
