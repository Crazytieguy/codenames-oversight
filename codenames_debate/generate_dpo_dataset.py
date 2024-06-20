from pathlib import Path

import typer
from pydantic import NonNegativeFloat, NonNegativeInt
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import InferenceSample
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

CLUE_DATASET: Path


@app.callback()
def set_clue_dataset(clue_dataset: Path):
    global CLUE_DATASET
    CLUE_DATASET = clue_dataset


def main(
    overseer: OverSeer,
):
    "Generate a DPO dataset from a dataset of clue pairs"
    data = [
        InferenceSample.model_validate_json(line)
        for line in CLUE_DATASET.read_text().splitlines()
    ]
    for sample in tqdm(data, desc="Overseeing"):
        evaluations = [evaluate_clue(sample.game, c) for c in sample.clue_critiques]
        oversights = [overseer.oversee(e) for e in evaluations]
        preference_set = PreferenceSet(
            game=sample.game, overseer=overseer, oversights=oversights
        )
        print(preference_set.model_dump_json())


@app.command()
def robust():
    overseer = RobustOverSeer()
    main(overseer)


@app.command()
def robust_judge():
    overseer = RobustJudgeOverSeer()
    main(overseer)


@app.command()
def neglect_last_n(neglect_words: NonNegativeInt):
    overseer = NeglectLastNOverSeer(neglect_words=neglect_words)
    main(overseer)


@app.command()
def negligent_biased(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
):
    overseer = NegligentBiasedOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
    )
    main(overseer)


if __name__ == "__main__":
    app()
