set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

neglect_words=(           0    2    0    0    1    0    2    0    0    1  )
bias_neglected_words=(    0    0    0    0    1    0    0    0    0    1  )
bias_non_neglected_words=(0    0    2    1    1    0    0    2    1    1  )
bias_factor=(             1.00 1.00 0.5  1.2  1.2  1.00 1.00 0.5  1.2  1.2)
adversarial_alpha=(       0.0  0.0  0.0  0.0  0.0  0.2  0.2  0.2  0.2  0.2)

neglect_words=${neglect_words[$array_task_id]}
bias_neglected_words=${bias_neglected_words[$array_task_id]}
bias_non_neglected_words=${bias_non_neglected_words[$array_task_id]}
bias_factor=${bias_factor[$array_task_id]}
adversarial_alpha=${adversarial_alpha[$array_task_id]}

>&2 echo "neglect_words: $neglect_words"
>&2 echo "bias_neglected_words: $bias_neglected_words"
>&2 echo "bias_non_neglected_words: $bias_non_neglected_words"
>&2 echo "bias_factor: $bias_factor"
>&2 echo "adversarial_alpha: $adversarial_alpha"

params_str=nw-$neglect_words-bnw-$bias_neglected_words-bnnw-$bias_non_neglected_words-bf-$bias_factor-aa-$adversarial_alpha

python -m codenames_oversight.rloo --adversarial-alpha $adversarial_alpha data/rloo-6-4-50400-dataset.jsonl $sft_model models/base-cluer-6-4-peft-with-vocab negligent-biased-base $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor

results_folder=results/rloo-6-4-negligent-biased-base
mkdir -p $results_folder
python -m codenames_oversight.evaluate_model --adversarial-alpha $adversarial_alpha $models/rloo-negligent-biased-base-6-4/$params_str/cluer negligent-biased-base $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor < data/eval-games-6-4.jsonl  > $results_folder/$params_str.jsonl
