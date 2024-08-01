set -e
source /ext3/env.sh
export $(cat .env)

task_id=$1

neglect_words=(1 0 0 1 1 0 0 1)
bias_neglected_words=(0 0 0 1 0 0 0 1)
bias_non_neglected_words=(0 1 1 1 0 1 1 1)
bias_factor=(1.0 0.64 1.5625 1.5625 1.0 0.64 1.5625 1.5625)
adversarial_alpha=(0.0 0.0 0.0 0.0 0.2 0.2 0.2 0.2)

neglect_words=${neglect_words[$task_id]}
bias_neglected_words=${bias_neglected_words[$task_id]}
bias_non_neglected_words=${bias_non_neglected_words[$task_id]}
bias_factor=${bias_factor[$task_id]}
adversarial_alpha=${adversarial_alpha[$task_id]}

python -m codenames_debate.ppo data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_negligent_biased/nw-$neglect_words-bnw-$bias_neglected_words-bnnw-$bias_non_neglected_words-bf-$bias_factor negligent_biased $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor
