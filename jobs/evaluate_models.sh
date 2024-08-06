set -e
# source /ext3/env.sh
export $(cat .env)

rm -f results/rloo-negligent-biased.jsonl

for folder in models/rloo_negligent_biased/*; do
  nw=${folder#*/nw-}
  nw=${nw%%-*}
  bnw=${folder#*-bnw-}
  bnw=${bnw%%-*}
  bnnw=${folder#*-bnnw-}
  bnnw=${bnnw%%-*}
  bf=${folder#*-bf-}
  bf=${bf%%-*}
  aa=${folder#*-aa-}

  >&2 echo "neglect_words: $nw"
  >&2 echo "bias_neglected_words: $bnw"
  >&2 echo "bias_non_neglected_words: $bnnw"
  >&2 echo "bias_factor: $bf"
  >&2 echo "adversarial_alpha: $aa"

  python -m codenames_debate.evaluate_model --adversarial-alpha $aa $folder negligent-biased $nw $bnw $bnnw $bf < data/eval-games-small.jsonl >> results/rloo-negligent-biased.jsonl
done
