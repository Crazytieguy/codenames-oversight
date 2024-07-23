set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1
calibrated_p_2=$2

python -m codenames_debate.ppo --learning-rate $learning_rate --calibrated-p-2 $calibrated_p_2 data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_small_robust_calibrated_2/lr-$learning_rate-p2-$calibrated_p_2 robust
