import logging
import sys

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


def main(clue_words: int = 2048):
    games = [Game.model_validate_json(line) for line in sys.stdin]
    all_overseers = [
        overseer_class(
            neglect_words=neglect_words,
            bias_neglected_words=bias_neglected_words,
            bias_non_neglected_words=bias_non_neglected_words,
            bias_factor=bias_factor,
        )
        for overseer_class in [
            NegligentBiasedJudgeOverSeer,
            # NegligentBiasedOverSeer
        ]
        for neglect_words in [
            # 0,
            1
        ]
        for bias_neglected_words in {
            # 0,
            neglect_words
        }
        for bias_non_neglected_words in [
            0,
            # 1
        ]
        for bias_factor in (
            [1.0]
            if (bias_neglected_words + bias_non_neglected_words == 0)
            else [0.64, 0.8, 1.25, 1.5625]
        )
    ]
    for overseer in tqdm(all_overseers, desc="Running negligent biased sweep"):
        for p_set in run_params(games, overseer, clue_words):
            print(p_set.model_dump_json())


def run_params(games: list[Game], overseer: OverSeer, clue_words: int):
    return [
        p_set
        for game in tqdm(games)
        for p_set in gen_optimal_preference_sets(overseer, game, clue_words)
    ]


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


if __name__ == "__main__":
    typer.run(main)
