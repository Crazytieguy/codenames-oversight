import json
import logging
import random
from itertools import cycle
from pathlib import Path
from typing import Optional

import typer
from pydantic import NonNegativeFloat, NonNegativeInt
from toolz.itertoolz import partition
from tqdm import tqdm

from .generate_random_sft_dataset import random_critique, random_targets
from .models import Clue, SFTSample, generate_game
from .oversight import (
    CLUE_WORDS_INDEXABLE,
    NeglectLastNOverSeer,
    NegligentBiasedBaseOverSeer,
    NegligentBiasedJudgeOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


VOCAB_FILE: Optional[str]
DATASET_SIZE: int
N_GOOD_WORDS: int
N_BAD_WORDS: int
OPTIMIZATION_STRENGTH: int


@app.callback()
def set_params(
    vocab_file: Optional[str] = None,
    dataset_size: int = 8192,
    n_good_words: int = 6,
    n_bad_words: int = 4,
    optimization_strength: int = 1,
):
    global VOCAB_FILE
    global DATASET_SIZE
    global N_GOOD_WORDS
    global N_BAD_WORDS
    global OPTIMIZATION_STRENGTH
    VOCAB_FILE = vocab_file
    DATASET_SIZE = dataset_size
    N_GOOD_WORDS = n_good_words
    N_BAD_WORDS = n_bad_words
    OPTIMIZATION_STRENGTH = optimization_strength


def main(overseer: OverSeer):
    random.seed(42)

    if VOCAB_FILE is not None:
        # Dict from clue word to count
        vocab: dict[str, int] = json.loads(Path(VOCAB_FILE).read_text())
        clue_words = [w for w, count in vocab.items() for _ in range(count)]
    else:
        clue_words = CLUE_WORDS_INDEXABLE

    random.shuffle(clue_words)
    clue_groups = partition(OPTIMIZATION_STRENGTH, cycle(clue_words))

    total_targets = 0
    for _ in tqdm(range(DATASET_SIZE), desc="Generating SFT dataset from overseer"):
        game = generate_game(N_GOOD_WORDS, N_BAD_WORDS)
        tries = 5
        while True:
            group = next_or_raise(clue_groups)
            try:
                clue_critiques = overseer.optimal(game, group, [0.0])[0.0]
                break
            except ValueError as e:
                logger.warning("Overseer failed to generate an optimal clue, retrying", exc_info=e)
                tries -= 1
                if tries == 0:
                    raise
        total_targets += len(clue_critiques.clue.targets)
        clue_word = clue_critiques.clue.clue
        targets = (
            []  # base overseer doesn't use targets
            if isinstance(overseer, NegligentBiasedBaseOverSeer)
            else random_targets(game)
        )
        clue = Clue(clue=clue_word, targets=targets)
        critique = random_critique(game, clue)
        sample = SFTSample(game=game, clue=clue, critique=critique)
        print(sample.model_dump_json())
    logger.info(f"Average number of oracle targets: {total_targets / DATASET_SIZE}")


def next_or_raise(iterator):
    v = next(iterator, None)
    if v is None:
        raise ValueError("Iterator exhausted")
    return v


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
    neglect_good_words: NonNegativeInt,
):
    overseer = NegligentBiasedOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
        neglect_good_words=neglect_good_words,
    )
    main(overseer)


@app.command()
def negligent_biased_judge(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
    neglect_good_words: NonNegativeInt,
):
    overseer = NegligentBiasedJudgeOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
        neglect_good_words=neglect_good_words,
    )
    main(overseer)


@app.command()
def negligent_biased_base(
    neglect_words: NonNegativeInt,
    bias_neglected_words: NonNegativeInt,
    bias_non_neglected_words: NonNegativeInt,
    bias_factor: NonNegativeFloat,
    neglect_good_words: NonNegativeInt,
):
    overseer = NegligentBiasedBaseOverSeer(
        neglect_words=neglect_words,
        bias_neglected_words=bias_neglected_words,
        bias_non_neglected_words=bias_non_neglected_words,
        bias_factor=bias_factor,
        neglect_good_words=neglect_good_words,
    )
    main(overseer)


if __name__ == "__main__":
    app()
