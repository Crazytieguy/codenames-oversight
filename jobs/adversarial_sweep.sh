set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

adversarial_alphas=(      0.0 0.05 0.1 0.2 0.3 0.0 0.05 0.1 0.2 0.3)
bias_non_neglected_words=(0   0    0   0   0   2   2    2   2   2  )
bias_factor=(             1.0 1.0  1.0 1.0 1.0 0.5 0.5  0.5 0.5 0.5)

adversarial_alpha=${adversarial_alphas[$array_task_id]}
bias_non_neglected_words=${bias_non_neglected_words[$array_task_id]}
bias_factor=${bias_factor[$array_task_id]}
params_str=aa-$adversarial_alpha-bnnw-$bias_non_neglected_words-bf-$bias_factor

>&2 echo "adversarial_alpha: $adversarial_alpha"
>&2 echo "bias_non_neglected_words: $bias_non_neglected_words"
>&2 echo "bias_factor: $bias_factor"

python -m codenames_oversight.rloo --adversarial-alpha $adversarial_alpha data/rloo-6-4-50400-dataset.jsonl models/base-cluer-6-4-peft-with-vocab models/rloo-adversarial-6-4/$params_str negligent-biased 0 0 $bias_non_neglected_words $bias_factor 0

results_folder=results/rloo-adversarial-6-4
mkdir -p $results_folder
python -m codenames_oversight.evaluate_model --adversarial-alpha $adversarial_alpha $models/rloo-adversarial-6-4/$params_str/cluer negligent-biased 0 0 $bias_non_neglected_words $bias_factor 0 < data/eval-games-6-4.jsonl > $results_folder/$params_str.jsonl