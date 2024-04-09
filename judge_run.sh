source /ext3/env.sh
export $(cat .env)
experiment_name=judge-$(date "+%Y-%m-%d_%H-%M-%S")
data_dir=data/$experiment_name
models_dir=models/$experiment_name
mkdir -p $data_dir
mkdir -p $models_dir
cp -r models/llama-7b-cluer-with-targets $models_dir/cluer-0
cp -r models/llama-7b-random-critiquer $models_dir/critiquer-0
for i in {0..5}
do
    python -m codenames_debate.generate_clues $models_dir/cluer-$i --num-games=3072 --clues-per-game=3 > $data_dir/clues-$i.jsonl
    python -m codenames_debate.generate_critiques $models_dir/critiquer-$i $data_dir/clues-$i.jsonl --critiques-per-clue=$(((6 / ($i + 1)) + 1)) > $data_dir/clue-critiques-$i.jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/clue-critiques-$i.jsonl --overseer judge > $data_dir/preference-sets-$i.jsonl
    python -m codenames_debate.dpo_dataset_to_critique_sft_dataset $data_dir/preference-sets-$i.jsonl >> $data_dir/good-critiques.jsonl
    tail -n 8192 $data_dir/good-critiques.jsonl > $data_dir/good-critiques-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-sets-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1))
    python -m codenames_debate.sft $models_dir/critiquer-$((i + 1)) $data_dir/good-critiques-$i.jsonl --model-role critiquer
done
python -m codenames_debate.generate_clues $models_dir/cluer-6 --num-games=1024 --clues-per-game=1 > $data_dir/clues-6.jsonl
python -m codenames_debate.generate_critiques $models_dir/critiquer-6 $data_dir/clues-6.jsonl --critiques-per-clue=2 > $data_dir/clue-critiques-6.jsonl
python -m codenames_debate.generate_dpo_dataset $data_dir/clue-critiques-6.jsonl --overseer judge > $data_dir/preference-sets-6.jsonl
