set -e
source /ext3/env.sh
export $(cat .env)

learning_rate=$1
kl_coeff=$2

>&2 echo "critiquer_learning_rate: $learning_rate"
>&2 echo "critiquer_kl_divergence: $kl_coeff"

data_dir=data/double-rloo-robust-critiquer-sweep
mkdir -p $data_dir

python -m codenames_oversight.rloo --critiquer-learning-rate=$learning_rate --critiquer-kl-coeff=$kl_coeff --critique-model-dir models/base-critiquer-peft data/rloo-small-11200-dataset.jsonl models/sft/negligent-biased/cluer/nw-0-bnw-0-bnnw-0-bf-1.00-aa-0.0 models/double-rloo-robust-critiquer-sweep/lr-$learning_rate-kl-$kl_coeff robust > $data_dir/lr-$learning_rate-kl-$kl_coeff.jsonl
