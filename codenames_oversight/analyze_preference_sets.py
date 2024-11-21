import csv
import sys
from enum import Enum
from pathlib import Path

import typer
from dotenv import load_dotenv
from toolz import groupby, partition
from tqdm import tqdm

from .oversight import (
    Evaluation,
    NeglectLastNOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

load_dotenv()
app = typer.Typer()


class DataIn(str, Enum):
    EVAL = "eval"
    TRAIN = "train"


@app.command()
def main(dataset_files: list[Path], data_in: DataIn = DataIn.EVAL, num_steps: int = 336):
    columns = [
        "protocol",
        "overseer",
        "adversarial_alpha",
        "true_score",
        "overseer_accuracy",
        "exploitation",
        "reference_estimated_score",
        "overseer_estimated_score",
    ]
    if data_in == DataIn.EVAL:
        columns.append("optimization_strength")

    if data_in == DataIn.TRAIN:
        columns.append("comparisons_performed")

    writer = csv.DictWriter(sys.stdout, columns)
    writer.writeheader()
    pbar = tqdm(desc="Processing")

    for dataset_file in dataset_files:
        total_lines = 0
        if data_in == DataIn.TRAIN:
            with dataset_file.open() as f:
                total_lines = sum(1 for _ in f)

        super_batch = []
        overseer = None
        optimization_strength = None
        max_batch_size = total_lines // num_steps if data_in == DataIn.TRAIN else float("inf")

        with dataset_file.open() as f:
            for line in f:
                p_set = PreferenceSet.model_validate_json(line)
                if overseer is None:
                    overseer = p_set.overseer
                    optimization_strength = p_set.optimization_strength

                new_batch = (overseer, optimization_strength) != (p_set.overseer, p_set.optimization_strength) or len(
                    super_batch
                ) >= max_batch_size

                if new_batch:
                    batches = groupby(lambda p_set: p_set.adversarial_alpha, super_batch)
                    rows = [
                        row
                        for adversarial_alpha, batch in batches.items()
                        for row in process_batch(
                            overseer, optimization_strength, adversarial_alpha, batch, data_in, num_steps
                        )
                    ]
                    pbar.update(1)
                    for row in rows:
                        writer.writerow(row)
                    super_batch = []
                    overseer = p_set.overseer
                    optimization_strength = p_set.optimization_strength
                super_batch.append(p_set)

        assert overseer is not None
        batches = groupby(lambda p_set: p_set.adversarial_alpha, super_batch)
        for adversarial_alpha, batch in batches.items():
            rows = process_batch(overseer, optimization_strength, adversarial_alpha, batch, data_in, num_steps)
            pbar.update(1)
            for row in rows:
                writer.writerow(row)

    pbar.close()


def process_batch(
    overseer: OverSeer,
    optimization_strength: int | None,
    adversarial_alpha: float,
    batch: list[PreferenceSet],
    data_in: DataIn,
    num_steps: int,
):
    if data_in == DataIn.TRAIN:
        batch_size = len(batch) // num_steps
        return [
            compute_row(data_in, overseer, optimization_strength, adversarial_alpha, p_sets)
            for p_sets in tqdm(partition(batch_size, batch), total=num_steps, desc="Analyzing")
        ]
    return [compute_row(data_in, overseer, optimization_strength, adversarial_alpha, batch)]


def compute_row(
    data_in: DataIn,
    overseer: OverSeer,
    optimization_strength: int | None,
    adversarial_alpha: float,
    p_sets: list[PreferenceSet],
) -> dict:
    assert not isinstance(overseer, (RobustOverSeer, RobustJudgeOverSeer, NeglectLastNOverSeer))
    overseer_name = {
        (0, 0, 0, 1.0, 0): "robust",
        (1, 0, 0, 1.0, 0): "weak-negligent",
        (2, 0, 0, 1.0, 0): "negligent",
        (0, 0, 2, 0.5, 0): "underweigh",
        (0, 0, 1, 1.2, 0): "overweigh",
        (0, 0, 0, 1.0, 2): "lazy",
        (1, 1, 0, 2.0, 0): "overlap",
    }[
        (
            overseer.neglect_words,
            overseer.bias_neglected_words,
            overseer.bias_non_neglected_words,
            overseer.bias_factor,
            overseer.neglect_good_words,
        )
    ]
    protocol = {
        "negligent_biased": "consultancy",
        "negligent_biased_base": "base",
        "negligent_biased_judge": "critiques",
    }[overseer.name]
    reference_overseer = overseer.reference_overseer()
    oversights = [o for p_set in p_sets for o in p_set.oversights]
    games_flat = [p_set.game for p_set in p_sets for _ in p_set.oversights]

    evaluations = [
        Evaluation(
            game=game,
            clue_critiques=o.clue_critiques,
            score=o.ground_truth_score,
            similarities=o.ground_truth_similarities,
        )
        for game, o in zip(games_flat, oversights)
    ]
    reference_oversights = [reference_overseer.oversee(e) for e in evaluations]
    mean_true_reward = sum(o.ground_truth_score for p_set in p_sets for o in p_set.oversights) / len(oversights)
    mean_reference_estimated_score = sum(
        reference_overseer.estimated_score(game, o) for game, o in zip(games_flat, reference_oversights)
    ) / len(reference_oversights)
    mean_overseer_estimated_score = sum(
        overseer.estimated_score(game, o) for game, o in zip(games_flat, oversights)
    ) / len(oversights)
    exploitation = mean_overseer_estimated_score - mean_reference_estimated_score
    overseer_accuracy = sum(
        o.expected_score == ro.expected_score for o, ro in zip(oversights, reference_oversights)
    ) / len(oversights)
    comparison_count = sum(o.comparisons_performed for o in oversights)
    row = {
        "protocol": protocol,
        "overseer": overseer_name,
        "adversarial_alpha": round(adversarial_alpha, 3),
        "exploitation": round(exploitation, 3),
        "overseer_accuracy": round(overseer_accuracy, 3),
        "true_score": round(mean_true_reward, 3),
        "reference_estimated_score": round(mean_reference_estimated_score, 3),
        "overseer_estimated_score": round(mean_overseer_estimated_score, 3),
    }
    if data_in == DataIn.EVAL:
        row["optimization_strength"] = optimization_strength
    if data_in == DataIn.TRAIN:
        row["comparisons_performed"] = comparison_count
    return row


if __name__ == "__main__":
    app()
