set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

neglect_words=(           0    2    0    0    1    0    2    0    0    1  )
bias_neglected_words=(    0    0    0    0    1    0    0    0    0    1  )
bias_non_neglected_words=(0    0    2    1    1    0    0    2    1    1  )
bias_factor=(             1.00 1.00 0.8  1.2  1.2  1.00 1.00 0.8  1.2  1.2)
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

sft_dataset=data/sft/negligent-biased-base/$params_str.jsonl
# if [ ! -e $sft_dataset ]; then
python -m codenames_oversight.generate_sft_from_overseer --vocab-file data/decent-vocab-mild.json negligent-biased-base $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor > $sft_dataset
# fi

sft_model=models/sft/negligent-biased-base/$params_str
# if [ ! -e $sft_model ]; then
python -m codenames_oversight.sft $sft_model $sft_dataset
# fi

python -m codenames_oversight.rloo --adversarial-alpha $adversarial_alpha data/rloo-small-22400-dataset.jsonl $sft_model models/rloo-small-negligent-biased/$params_str negligent-biased-base $neglect_words $bias_neglected_words $bias_non_neglected_words $bias_factor
