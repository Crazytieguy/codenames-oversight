set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

neglect_words=(           0    1    0    0    1    0    1    0    0    1   )
bias_neglected_words=(    0    0    0    0    1    0    0    0    0    1   )
bias_non_neglected_words=(0    0    1    1    1    0    0    1    1    1   )
bias_factor=(             1.00 1.00 0.64 1.56 1.56 1.00 1.00 0.64 1.56 1.56)
adversarial_alpha=(       0.0  0.0  0.0  0.0  0.0  0.2  0.2  0.2  0.2  0.2)

neglect_words=${neglect_words[$array_task_id]}
bias_neglected_words=${bias_neglected_words[$array_task_id]}
bias_non_neglected_words=${bias_non_neglected_words[$array_task_id]}
bias_factor=${bias_factor[$array_task_id]}
adversarial_alpha=${adversarial_alpha[$array_task_id]}

echo "neglect_words: $neglect_words"
echo "bias_neglected_words: $bias_neglected_words"
echo "bias_non_neglected_words: $bias_non_neglected_words"
echo "bias_factor: $bias_factor"
echo "adversarial_alpha: $adversarial_alpha"

python -m codenames_debate.ppo data/ppo-small-dataset.jsonl models/llama-7b-random-cluer-small-merged models/ppo_negligent_biased/nw-$neglect_words-bnw-$bias_neglected_words-bnnw-$bias_non_neglected_words-bf-$bias_factor negligent_biased $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor
