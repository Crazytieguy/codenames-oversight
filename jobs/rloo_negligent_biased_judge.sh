set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

neglect_words=(           0    2    0    0    1    0    0    2    0    0    1   0   )
bias_neglected_words=(    0    0    0    0    1    0    0    0    0    0    1   0   )
bias_non_neglected_words=(0    0    2    1    1    0    0    0    2    1    1   0   )
bias_factor=(             1.00 1.00 0.5  1.2  1.2  1.00 1.00 1.00 0.5  1.2  1.2 1.00)
neglect_good_words=(      0    0    0    0    0    2    0    0    0    0    0   2   )
adversarial_alpha=(       0.0  0.0  0.0  0.0  0.0  0.0  0.2  0.2  0.2  0.2  0.2 0.2 )

neglect_words=${neglect_words[$array_task_id]}
bias_neglected_words=${bias_neglected_words[$array_task_id]}
bias_non_neglected_words=${bias_non_neglected_words[$array_task_id]}
bias_factor=${bias_factor[$array_task_id]}
neglect_good_words=${neglect_good_words[$array_task_id]}
adversarial_alpha=${adversarial_alpha[$array_task_id]}

>&2 echo "neglect_words: $neglect_words"
>&2 echo "bias_neglected_words: $bias_neglected_words"
>&2 echo "bias_non_neglected_words: $bias_non_neglected_words"
>&2 echo "bias_factor: $bias_factor"
>&2 echo "neglect_good_words: $neglect_good_words"
>&2 echo "adversarial_alpha: $adversarial_alpha"

params_str=nw-$neglect_words-bnw-$bias_neglected_words-bnnw-$bias_non_neglected_words-bf-$bias_factor-ngw-$neglect_good_words-aa-$adversarial_alpha

python -m codenames_oversight.rloo --critique-model-dir models/base-critiquer-6-4-peft --adversarial-alpha $adversarial_alpha data/rloo-6-4-50400-dataset.jsonl models/base-cluer-6-4-peft-with-vocab models/rloo-negligent-biased-judge-6-4/$params_str negligent-biased $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor $neglect_good_words

results_folder=results/rloo-negligent-biased-judge-6-4
mkdir -p $results_folder
python -m codenames_oversight.evaluate_model --critique-adapter $models/rloo-negligent-biased-judge-6-4/$params_str/critiquer --adversarial-alpha $adversarial_alpha $models/rloo-negligent-biased-judge-6-4/$params_str/cluer negligent-biased-judge $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor $neglect_good_words < data/eval-games-6-4.jsonl > $results_folder/$params_str.jsonl
