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
    python -m codenames_debate.generate_clues $models_dir/cluer-$i --diversity-penalty=$(echo "scale=1; 2.0 - 0.3 * $i" | bc) > $data_dir/clues-$i.jsonl
    python -m codenames_debate.generate_critiques $models_dir/critiquer-$i $data_dir/clues-$i.jsonl --critiques-per-clue=$((7 - $i)) > $data_dir/clue-critiques-$i.jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/clue-critiques-$i.jsonl --overseer judge > $data_dir/preference-pairs-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-pairs-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1)) --reference-model $models_dir/cluer-0
    python -m codenames_debate.dpo_dataset_to_critique_sft_dataset $data_dir/preference-pairs-$i.jsonl >> $data_dir/good-critiques.jsonl
    python -m codenames_debate.sft $models_dir/critiquer-$((i + 1)) $data_dir/good-critiques.jsonl --model-role critiquer
done