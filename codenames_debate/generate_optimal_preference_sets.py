import sys
from typing import Annotated

import typer
from pydantic import NonNegativeFloat, NonNegativeInt
from tqdm import tqdm

from .evaluate_clue import evaluate_clue
from .models import Game
from .oversight import (
    NeglectLastNOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
    RobustJudgeOverSeer,
    RobustOverSeer,
)

app = typer.Typer()

ADVERSARIAL_ALPHA: float


@app.callback()
def set_adversarial_alpha(
    adversarial_alpha: Annotated[
        float,
        typer.Argument(
            help="How much to reward the model for performing poorly on the ground truth."
        ),
    ] = 0.0,
):
    global ADVERSARIAL_ALPHA
    ADVERSARIAL_ALPHA = adversarial_alpha


def main(overseer: OverSeer):
    """Generate optimal clues for a dataset of games, and provide the final PreferenceSets."""
    games = [Game.model_validate_json(line) for line in sys.stdin]
    for game in tqdm(games, desc="Generating optimal clues"):
        preference_set = gen_optimal_preference_set(overseer, game, ADVERSARIAL_ALPHA)
        print(preference_set.model_dump_json())


def gen_optimal_preference_set(
    overseer: OverSeer,
    game: Game,
    adversarial_alpha: float,
) -> PreferenceSet:
    clue_critiques = overseer.optimal(game, adversarial_alpha)
    oversight = overseer.oversee(evaluate_clue(game, clue_critiques))
    preference_set = PreferenceSet(
        game=game,
        overseer=overseer,
        oversights=[oversight],
        adversarial_alpha=adversarial_alpha,
    )
    return preference_set


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
