source /ext3/env.sh
export $(cat .env)
experiment_name=robust-$(date "+%Y-%m-%d_%H-%M-%S")
data_dir=data/$experiment_name
models_dir=models/$experiment_name
mkdir -p $data_dir
mkdir -p $models_dir
cp -r models/llama-7b-cluer-with-targets $models_dir/cluer-0
for i in {0..6}
do
    python -m codenames_debate.generate_clues $models_dir/cluer-$i --num-games=2048 --diversity-penalty=$(echo "scale=1; 1.8 - 0.2 * $i" | bc) > $data_dir/clues-$i.jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/clues-$i.jsonl --overseer robust > $data_dir/preference-pairs-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-pairs-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1))
done
python -m codenames_debate.generate_clues $models_dir/cluer-7 --num-games=1024 --clues-per-game=1 > $data_dir/clues-7.jsonl
python -m codenames_debate.generate_dpo_dataset $data_dir/clues-7.jsonl --overseer robust > $data_dir/preference-pairs-7.jsonl
