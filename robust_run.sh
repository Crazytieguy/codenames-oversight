set -e
source /ext3/env.sh
export $(cat .env)
games_per_phase=$1
experiment_name=robust-$games_per_phase-games-per-phase-$(date "+%Y-%m-%d_%H-%M-%S")
data_dir=data/$experiment_name
models_dir=models/$experiment_name
mkdir -p $data_dir
mkdir -p $models_dir
cp -r models/llama-7b-random-cluer $models_dir/cluer-0
cp data/training-games.jsonl $data_dir/
cp data/eval-clues-0.jsonl $data_dir/
python -m codenames_debate.generate_dpo_dataset $data_dir/eval-clues-0.jsonl --overseer robust > $data_dir/eval-preference-sets-0.jsonl
for i in {0..9}
do
    head -n $games_per_phase $data_dir/training-games.jsonl | python -m codenames_debate.generate_clues --model-name-or-path=$models_dir/cluer-$i --clues-per-game=6 --temperature=1.5 > $data_dir/clues-$i.jsonl
    # sed -i "1,${games_per_phase}d" $data_dir/training-games.jsonl
    sed -i "1,8192d" $data_dir/training-games.jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/clues-$i.jsonl --overseer robust > $data_dir/preference-sets-$i.jsonl
    python -m codenames_debate.dpo $data_dir/preference-sets-$i.jsonl $models_dir/cluer-$i $models_dir/cluer-$((i + 1))
    python -m codenames_debate.generate_clues --model-name-or-path=$models_dir/cluer-$((i + 1)) --clues-per-game=1 < data/evaluation-games.jsonl > $data_dir/eval-clues-$((i + 1)).jsonl
    python -m codenames_debate.generate_dpo_dataset $data_dir/eval-clues-$((i + 1)).jsonl --overseer robust > $data_dir/eval-preference-sets-$((i + 1)).jsonl
done
