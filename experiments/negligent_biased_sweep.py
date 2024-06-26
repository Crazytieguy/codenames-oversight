import logging
import sys
from typing import Annotated

import typer
from tqdm import tqdm

from codenames_debate.evaluate_clue import evaluate_clue
from codenames_debate.models import Game
from codenames_debate.oversight import (
    NegligentBiasedJudgeOverSeer,
    NegligentBiasedOverSeer,
    OverSeer,
    PreferenceSet,
)

logging.basicConfig(level=logging.INFO)


def main(debate: Annotated[bool, typer.Option()], clue_words: int = 1024):
    games = [Game.model_validate_json(line) for line in sys.stdin]
    overseer_class = NegligentBiasedJudgeOverSeer if debate else NegligentBiasedOverSeer
    all_optimization_params = [
        overseer_class(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
        )
        for neglect_words in [0, 1]
        for bias_neglected_words in {0, neglect_words}
        for bias_non_neglected_words in [0, 1]
        for bias_factor in (
            [1.0]
            if (bias_neglected_words + bias_non_neglected_words == 0)
            else [0.7, 1.5]
        )
    ]
    for overseer in tqdm(
        all_optimization_params, desc="Running negligent biased sweep"
    ):
        for p_set in run_params(games, overseer, clue_words):
            print(p_set.model_dump_json())


def gen_optimal_preference_sets(
    overseer: OverSeer,
    game: Game,
    clue_words: int,
) -> list[PreferenceSet]:
    clue_critiques_by_alpha = overseer.optimal(
        game, clue_words, [i * 0.05 for i in range(10)]
    )
    oversights = {
        adversarial_alpha: overseer.oversee(evaluate_clue(game, clue_critiques))
        for adversarial_alpha, clue_critiques in clue_critiques_by_alpha.items()
    }
    return [
        PreferenceSet(
            game=game,
            overseer=overseer,
            oversights=[oversight],
            adversarial_alpha=adversarial_alpha,
        )
        for adversarial_alpha, oversight in oversights.items()
    ]


def run_params(games: list[Game], overseer: OverSeer, clue_words: int):
    return [
        p_set
        for game in tqdm(games)
        for p_set in gen_optimal_preference_sets(overseer, game, clue_words)
    ]


if __name__ == "__main__":
    typer.run(main)
