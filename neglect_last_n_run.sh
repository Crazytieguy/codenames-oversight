set -e
source /ext3/env.sh
export $(cat .env)
neglect_words=$1
echo "Neglecting last $neglect_words words"
experiment_name=neglect-last-$neglect_words-$(date "+%Y-%m-%d_%H-%M-%S")
data_dir=data/$experiment_name
models_dir=models/$experiment_name
mkdir -p $data_dir
mkdir -p $models_dir
cp -r models/llama-7b-cluer-with-targets $models_dir/cluer-0
cp data/clue-triples-0.jsonl $data_dir/clues-0.jsonl
for i in {0..3}
do
    if [ "$i" -ne 0 ]; then
        python -m codenames_debate.generate_clues $models_dir/cluer-$i --num-games=2048 --clues-per-game=3 > $data_dir/clues-$i.jsonl
    fi
    python -m codenames_debate.generate_dpo_dataset $data_dir/clues-$i.jsonl --concurrency=8 --overseer neglect_last_n --neglect-words $neglect_words > $data_dir/preference-pairs-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-pairs-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1))
done
python -m codenames_debate.generate_clues $models_dir/cluer-4 --num-games=1024 --clues-per-game=1 > $data_dir/clues-4.jsonl
python -m codenames_debate.generate_dpo_dataset $data_dir/clues-4.jsonl --concurrency=8 --overseer neglect_last_n --neglect-words $neglect_words > $data_dir/preference-pairs-4.jsonl
