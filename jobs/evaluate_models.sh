set -e
source /ext3/env.sh
export $(cat .env)

array_task_id=$1

folders=(models/rloo-negligent-biased-6-4/*)
folder=${folders[$array_task_id]}

nw=${folder#*/nw-}
nw=${nw%%-*}
bnw=${folder#*-bnw-}
bnw=${bnw%%-*}
bnnw=${folder#*-bnnw-}
bnnw=${bnnw%%-*}
bf=${folder#*-bf-}
bf=${bf%%-*}
ngw=${folder#*-ngw-}
ngw=${ngw%%-*}
aa=${folder#*-aa-}

>&2 echo "neglect_words: $nw"
>&2 echo "bias_neglected_words: $bnw"
>&2 echo "bias_non_neglected_words: $bnnw"
>&2 echo "bias_factor: $bf"
>&2 echo "neglect_good_words: $ngw"
>&2 echo "adversarial_alpha: $aa"

results_folder=results/rloo-negligent-biased-6-4
param_str=nw-$nw-bnw-$bnw-bnnw-$bnnw-bf-$bf-ngw-$ngw-aa-$aa
python -m codenames_oversight.evaluate_model --adversarial-alpha $aa $folder/cluer negligent-biased $nw $bnw $bnnw $bf $ngw < data/eval-games-6-4.jsonl > $results_folder/$param_str.jsonl
