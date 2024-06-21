import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
from tqdm import tqdm

from codenames_debate.generate_optimal_preference_sets import gen_optimal_preference_set
from codenames_debate.models import Game
from codenames_debate.oversight import (
    NegligentBiasedOverSeer,
    OverSeer,
)


def main():
    games = [Game.model_validate_json(line) for line in sys.stdin]
    all_optimization_params = [
        (
            NegligentBiasedOverSeer(
                neglect_words=neglect_words,
                bias_neglected_words=bias_neglected_words,
                bias_non_neglected_words=bias_non_neglected_words,
                bias_factor=bias_factor,
            ),
            adversarial_alpha,
        )
        for neglect_words in [0, 1]
        for bias_neglected_words in {0, neglect_words}
        for bias_non_neglected_words in [0, 1]
        for bias_factor in (
            [1.0]
            if (bias_neglected_words + bias_non_neglected_words == 0)
            else [0.7, 1.5]
        )
        for adversarial_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_params, games, overseer, adversarial_alpha)
            for overseer, adversarial_alpha in all_optimization_params
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running negligent biased sweep",
        ):
            for p_set in future.result():
                print(p_set.model_dump_json())


def run_params(games: list[Game], overseer: OverSeer, adversarial_alpha: float):
    return [
        gen_optimal_preference_set(overseer, game, adversarial_alpha) for game in games
    ]


if __name__ == "__main__":
    typer.run(main)
