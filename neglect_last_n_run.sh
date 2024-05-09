set -e
source /ext3/env.sh
export $(cat .env)
neglect_words=$1
adversarial_alpha=${2:-0.0}
echo "Neglecting last $neglect_words words"
echo "Adversarial alpha: $adversarial_alpha"
experiment_name=neglect-last-$neglect_words-alpha-$adversarial_alpha-$(date "+%Y-%m-%d_%H-%M-%S")
data_dir=data/$experiment_name
models_dir=models/$experiment_name
mkdir -p $data_dir
mkdir -p $models_dir
cp -r models/llama-7b-cluer-with-targets $models_dir/cluer-0
python -m codenames_debate.generate_clues $models_dir/cluer-0 --num-games=512 --clues-per-game=1 > $data_dir/eval-clues-0.jsonl
python -m codenames_debate.generate_dpo_dataset $data_dir/eval-clues-0.jsonl --concurrency=16 --overseer neglect_last_n --neglect-words $neglect_words > $data_dir/eval-preference-sets-0.jsonl
for i in {0..4}
do
    python -m codenames_debate.generate_clues $models_dir/cluer-$i --num-games=3072 --clues-per-game=3 --diversity-penalty=$(echo "1.6 - $i * 0.15" | bc) > $data_dir/clues-$i.jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/clues-$i.jsonl --concurrency=16 --overseer neglect_last_n --neglect-words $neglect_words > $data_dir/preference-sets-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-sets-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1)) $adversarial_alpha
    python -m codenames_debate.generate_clues $models_dir/cluer-$((i + 1)) --num-games=512 --clues-per-game=1 > $data_dir/eval-clues-$((i + 1)).jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/eval-clues-$((i + 1)).jsonl --concurrency=16 --overseer neglect_last_n --neglect-words $neglect_words > $data_dir/eval-preference-sets-$((i + 1)).jsonl
done
