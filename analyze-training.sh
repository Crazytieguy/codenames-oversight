python -m codenames_oversight.analyze_preference_sets data/negligent-biased-base/robust.jsonl --data-in train > results/training/negligent-biased-base/robust.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased-base/negligent.jsonl --data-in train > results/training/negligent-biased-base/negligent.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased-base/negligent-adversarial.jsonl --data-in train > results/training/negligent-biased-base/negligent-adversarial.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased/robust.jsonl --num-steps 672 --data-in train > results/training/negligent-biased/robust.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased/negligent.jsonl --num-steps 672 --data-in train > results/training/negligent-biased/negligent.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased/negligent-adversarial.jsonl --num-steps 672 --data-in train > results/training/negligent-biased/negligent-adversarial.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased-judge/robust.jsonl --num-steps 672 --data-in train > results/training/negligent-biased-judge/robust.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased-judge/underweigh.jsonl --num-steps 672 --data-in train > results/training/negligent-biased-judge/underweigh.csv
python -m codenames_oversight.analyze_preference_sets data/negligent-biased-judge/underweigh-adversarial.jsonl --num-steps 672 --data-in train > results/training/negligent-biased-judge/underweigh-adversarial.csv
